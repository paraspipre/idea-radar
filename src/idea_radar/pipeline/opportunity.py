"""Stages 5-8: competition research, sourced sizing, skeptic review, opportunity ranking.

Only the top-N clusters by demand get the expensive stages, and only when their analysis is
missing, older than 30 days, or the cluster grew >25% since. Stage 5 is the only stage with
web tools; it also records market facts (URL + exact quote) that stage 6 is allowed to cite.
Importing this module registers the stages on the pipeline graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from idea_radar.clock import utc_now
from idea_radar.db.models import (
    Analysis,
    AnalysisKind,
    Cluster,
    ClusterStatus,
    DemandScore,
    MarketFact,
    Opportunity,
)
from idea_radar.llm.errors import LLMOutputError, LLMRefusalError, StopRun
from idea_radar.llm.prompts import load_prompt
from idea_radar.llm.types import LLMRequest
from idea_radar.log import get_logger
from idea_radar.pipeline.cluster import cluster_members_text
from idea_radar.pipeline.graph import EXTRA_STAGES, PipelineContext
from idea_radar.scoring.opportunity import (
    Feasibility,
    SizingInput,
    competition_score,
    compute_sizing,
    feasibility_score,
    monetization_score,
    opportunity_score,
)

_log = get_logger(component="opportunity")

STALE_AFTER = timedelta(days=30)
REGROW = 0.25

# ------------------------------------------------------------------ schemas


class Competitor(BaseModel):
    name: str
    url: str
    positioning: str
    pricing: str | None = None
    complaints: list[str] = Field(default_factory=list)


class FoundFact(BaseModel):
    claim: str
    value: float | None = None
    unit: str | None = None
    source_url: str
    quote: str


class CompetitionResult(BaseModel):
    competitors: list[Competitor]
    crowdedness: float = Field(ge=0, le=1)
    dissatisfaction: float = Field(ge=0, le=1)
    rationale: str
    facts: list[FoundFact] = Field(default_factory=list)


class SizingResult(BaseModel):
    population: SizingInput
    share_with_problem: SizingInput
    price_per_year: SizingInput
    reachable_share: SizingInput
    obtainable_share: SizingInput
    feasibility: Feasibility
    rationale: str


class Objection(BaseModel):
    claim: str
    severity: Literal["low", "medium", "high"]


class SkepticResult(BaseModel):
    verdict: Literal["kill", "weak", "survives"]
    objections: list[Objection]
    what_would_change_my_mind: str
    confidence: Literal["low", "medium", "high"]


# ------------------------------------------------------------------ selection


@dataclass(frozen=True)
class Brief:
    cluster_id: int
    name: str
    summary: str
    who: str
    size: int
    demand: float
    demand_components: dict[str, Any]
    examples: list[dict[str, Any]]

    @property
    def wtp(self) -> float:
        normalized = self.demand_components.get("normalized", {})
        return float(normalized.get("wtp", 0.0))


def top_briefs(session: Session, ctx: PipelineContext, n: int) -> list[Brief]:
    latest: dict[int, DemandScore] = {}
    for score in session.scalars(select(DemandScore).order_by(DemandScore.created_at)):
        latest[score.cluster_id] = score
    clusters = {
        c.id: c
        for c in session.scalars(
            select(Cluster).where(
                Cluster.status == ClusterStatus.ACTIVE.value, Cluster.name.is_not(None)
            )
        )
    }
    ranked = sorted(
        (cid for cid in clusters if cid in latest), key=lambda c: latest[c].score, reverse=True
    )[:n]
    return [
        Brief(
            cluster_id=cid,
            name=clusters[cid].name or "",
            summary=clusters[cid].summary or "",
            who=clusters[cid].who or "",
            size=clusters[cid].size,
            demand=latest[cid].score,
            demand_components=latest[cid].components,
            examples=cluster_members_text(ctx.factory, cid)[:5],
        )
        for cid in ranked
    ]


def latest_analysis(session: Session, cluster_id: int, kind: AnalysisKind) -> Analysis | None:
    return session.scalars(
        select(Analysis)
        .where(Analysis.cluster_id == cluster_id, Analysis.kind == kind.value)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(1)
    ).first()


def is_stale(analysis: Analysis | None, size: int, *, after: Analysis | None = None) -> bool:
    """Missing, old, cluster grew materially, or older than the analysis it depends on."""
    if analysis is None:
        return True
    if utc_now() - analysis.created_at > STALE_AFTER:
        return True
    if size > analysis.cluster_size * (1 + REGROW):
        return True
    return after is not None and (after.created_at, after.id) > (
        analysis.created_at,
        analysis.id,
    )


def brief_text(brief: Brief) -> str:
    lines = [
        f"Problem: {brief.name}",
        f"Who: {brief.who}",
        f"Summary: {brief.summary}",
        f"Evidence: {brief.size} posts. Examples:",
    ]
    for ex in brief.examples:
        lines.append(f"- [{ex['source']}] {ex['who']}: {ex['problem']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ stages


@dataclass
class StageCounts:
    candidates: int = 0
    analysed: int = 0
    fresh: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


async def stage_competition(ctx: PipelineContext) -> dict[str, Any]:
    prompt = load_prompt("competition", 1)
    counts = StageCounts()
    with ctx.factory() as session:
        briefs = top_briefs(session, ctx, ctx.settings.demand.top_n)
        todo = [
            b
            for b in briefs
            if is_stale(latest_analysis(session, b.cluster_id, AnalysisKind.COMPETITION), b.size)
        ]
    counts.candidates, counts.fresh = len(briefs), len(briefs) - len(todo)
    for brief in todo:
        request: LLMRequest[CompetitionResult] = LLMRequest(
            stage="competition",
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            system=prompt.text,
            prompt=brief_text(brief),
            schema=CompetitionResult,
            tier="strong",
            tools=("WebSearch", "WebFetch"),
            max_turns=14,
        )
        try:
            result = (await ctx.client.complete_json(request)).data
        except StopRun as exc:
            ctx.stop = exc
            break
        except (LLMOutputError, LLMRefusalError) as exc:
            _log.error("competition_failed", cluster=brief.cluster_id, error=str(exc))
            counts.failed += 1
            continue
        with ctx.factory() as session:
            fact_ids = []
            for fact in result.facts:
                if not fact.source_url.startswith(("http://", "https://")) or not fact.quote:
                    continue
                row = MarketFact(
                    cluster_id=brief.cluster_id,
                    claim=fact.claim,
                    value=fact.value,
                    unit=fact.unit,
                    source_url=fact.source_url,
                    quote=fact.quote,
                )
                session.add(row)
                session.flush()
                fact_ids.append(row.id)
            payload = result.model_dump()
            payload["fact_ids"] = fact_ids
            payload["score"] = competition_score(result.crowdedness, result.dissatisfaction)
            session.add(
                Analysis(
                    cluster_id=brief.cluster_id,
                    kind=AnalysisKind.COMPETITION.value,
                    run_id=ctx.run_id,
                    cluster_size=brief.size,
                    payload=payload,
                )
            )
            session.commit()
        counts.analysed += 1
    return counts.as_dict()


def facts_text(facts: list[MarketFact]) -> str:
    if not facts:
        return "Facts: none found. Every input must be an explicit assumption."
    lines = ["Facts you may cite (by id):"]
    for f in facts:
        value = f" ({f.value:g} {f.unit or ''})".rstrip() if f.value is not None else ""
        lines.append(f'[F{f.id}] {f.claim}{value}. Source: {f.source_url} Quote: "{f.quote}"')
    return "\n".join(lines)


async def stage_sizing(ctx: PipelineContext) -> dict[str, Any]:
    prompt = load_prompt("sizing", 1)
    counts = StageCounts()
    jobs: list[tuple[Brief, Analysis, list[MarketFact]]] = []
    with ctx.factory() as session:
        briefs = top_briefs(session, ctx, ctx.settings.demand.top_n)
        counts.candidates = len(briefs)
        for b in briefs:
            competition = latest_analysis(session, b.cluster_id, AnalysisKind.COMPETITION)
            sizing = latest_analysis(session, b.cluster_id, AnalysisKind.SIZING)
            if competition is None:
                continue
            if not is_stale(sizing, b.size, after=competition):
                counts.fresh += 1
                continue
            ids = [int(i) for i in competition.payload.get("fact_ids", [])]
            facts = list(session.scalars(select(MarketFact).where(MarketFact.id.in_(ids))))
            jobs.append((b, competition, facts))
    for brief, competition, facts in jobs:
        pricing = "; ".join(
            f"{c['name']}: {c.get('pricing') or 'pricing unknown'}"
            for c in competition.payload.get("competitors", [])
        )
        request: LLMRequest[SizingResult] = LLMRequest(
            stage="sizing",
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            system=prompt.text,
            prompt=f"{brief_text(brief)}\nCompetitor pricing: {pricing or 'none found'}\n"
            f"{facts_text(facts)}",
            schema=SizingResult,
            tier="strong",
        )
        try:
            result = (await ctx.client.complete_json(request)).data
        except StopRun as exc:
            ctx.stop = exc
            break
        except (LLMOutputError, LLMRefusalError) as exc:
            _log.error("sizing_failed", cluster=brief.cluster_id, error=str(exc))
            counts.failed += 1
            continue
        totals = compute_sizing(
            {
                "population": result.population,
                "share_with_problem": result.share_with_problem,
                "price_per_year": result.price_per_year,
                "reachable_share": result.reachable_share,
                "obtainable_share": result.obtainable_share,
            },
            {f"F{f.id}" for f in facts},
        )
        fact_urls = {f"F{f.id}": f.source_url for f in facts}
        payload = {
            "inputs": {
                k: {**asdict(v), "source_url": fact_urls.get(v.fact_id or "")}
                for k, v in totals.inputs.items()
            },
            "tam": totals.tam,
            "sam": totals.sam,
            "som": totals.som,
            "sourced_share": totals.sourced_share,
            "monetization": monetization_score(totals.som[1], brief.wtp),
            "feasibility": feasibility_score(result.feasibility),
            "feasibility_detail": result.feasibility.model_dump(),
            "rationale": result.rationale,
        }
        with ctx.factory() as session:
            session.add(
                Analysis(
                    cluster_id=brief.cluster_id,
                    kind=AnalysisKind.SIZING.value,
                    run_id=ctx.run_id,
                    cluster_size=brief.size,
                    payload=payload,
                )
            )
            session.commit()
        counts.analysed += 1
    return counts.as_dict()


def money(value: float) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if value >= scale:
            return f"${value / scale:.1f}{unit}"
    return f"${value:,.0f}"


async def stage_skeptic(ctx: PipelineContext) -> dict[str, Any]:
    prompt = load_prompt("skeptic", 1)
    counts = StageCounts()
    jobs: list[tuple[Brief, Analysis, Analysis]] = []
    with ctx.factory() as session:
        briefs = top_briefs(session, ctx, ctx.settings.demand.top_n)
        counts.candidates = len(briefs)
        for b in briefs:
            competition = latest_analysis(session, b.cluster_id, AnalysisKind.COMPETITION)
            sizing = latest_analysis(session, b.cluster_id, AnalysisKind.SIZING)
            skeptic = latest_analysis(session, b.cluster_id, AnalysisKind.SKEPTIC)
            if competition is None or sizing is None:
                continue
            if not is_stale(skeptic, b.size, after=sizing):
                counts.fresh += 1
                continue
            jobs.append((b, competition, sizing))
    for brief, competition, sizing in jobs:
        comp, size = competition.payload, sizing.payload
        assumptions = "; ".join(
            f"{k}={v['base']:g} ({v['basis']}: {v.get('assumption') or v.get('fact_id')})"
            for k, v in size["inputs"].items()
        )
        evidence = (
            f"{brief_text(brief)}\n"
            f"Demand breakdown (0-1): {brief.demand_components.get('normalized', {})}\n"
            f"Competitors: {', '.join(c['name'] for c in comp.get('competitors', [])) or 'none'}"
            f" | crowdedness {comp['crowdedness']:.2f}, dissatisfaction "
            f"{comp['dissatisfaction']:.2f}\n"
            f"Sizing (base): TAM {money(size['tam'][1])}, SAM {money(size['sam'][1])}, "
            f"SOM {money(size['som'][1])}; {size['sourced_share']:.0%} of inputs sourced.\n"
            f"Inputs: {assumptions}"
        )
        request: LLMRequest[SkepticResult] = LLMRequest(
            stage="skeptic",
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            system=prompt.text,
            prompt=evidence,
            schema=SkepticResult,
            tier="strong",
        )
        try:
            result = (await ctx.client.complete_json(request)).data
        except StopRun as exc:
            ctx.stop = exc
            break
        except (LLMOutputError, LLMRefusalError) as exc:
            _log.error("skeptic_failed", cluster=brief.cluster_id, error=str(exc))
            counts.failed += 1
            continue
        with ctx.factory() as session:
            session.add(
                Analysis(
                    cluster_id=brief.cluster_id,
                    kind=AnalysisKind.SKEPTIC.value,
                    run_id=ctx.run_id,
                    cluster_size=brief.size,
                    payload=result.model_dump(),
                )
            )
            session.commit()
        counts.analysed += 1
    return counts.as_dict()


async def stage_rank(ctx: PipelineContext) -> dict[str, Any]:
    """Free: score every cluster that has demand + competition + sizing; store the ranking."""
    rows: list[dict[str, Any]] = []
    with ctx.factory() as session:
        latest: dict[int, DemandScore] = {}
        for score in session.scalars(select(DemandScore).order_by(DemandScore.created_at)):
            latest[score.cluster_id] = score
        active = set(
            session.scalars(select(Cluster.id).where(Cluster.status == ClusterStatus.ACTIVE.value))
        )
        for cid in active & set(latest):
            competition = latest_analysis(session, cid, AnalysisKind.COMPETITION)
            sizing = latest_analysis(session, cid, AnalysisKind.SIZING)
            if competition is None or sizing is None:
                continue
            skeptic = latest_analysis(session, cid, AnalysisKind.SKEPTIC)
            result = opportunity_score(
                demand=latest[cid].score,
                competition=float(competition.payload["score"]),
                monetization=float(sizing.payload["monetization"]),
                feasibility=float(sizing.payload["feasibility"]),
            )
            rows.append(
                {
                    "cluster_id": cid,
                    "result": result,
                    "competition": float(competition.payload["score"]),
                    "verdict": skeptic.payload["verdict"] if skeptic else None,
                }
            )
        rows.sort(key=lambda r: r["result"].score, reverse=True)
        for rank, row in enumerate(rows, start=1):
            result = row["result"]
            session.add(
                Opportunity(
                    cluster_id=row["cluster_id"],
                    run_id=ctx.run_id,
                    rank=rank,
                    score=result.score,
                    demand=result.factors["demand"],
                    competition=row["competition"],
                    monetization=result.factors["monetization"],
                    feasibility=result.factors["feasibility"],
                    verdict=row["verdict"],
                    breakdown={"factors": result.factors, "raw_product": result.raw_product},
                )
            )
        session.commit()
    return {"ranked": len(rows)}


def register() -> None:
    names = {name for name, _ in EXTRA_STAGES}
    for name, fn in (
        ("competition", stage_competition),
        ("sizing", stage_sizing),
        ("skeptic", stage_skeptic),
        ("rank", stage_rank),
    ):
        if name not in names:
            EXTRA_STAGES.append((name, fn))


register()


def latest_opportunities(factory: Any) -> list[dict[str, Any]]:
    """The most recent ranking (one rank run), joined with names and sizing totals."""
    with factory() as session:
        last = session.scalars(
            select(Opportunity).order_by(Opportunity.created_at.desc(), Opportunity.id.desc())
        ).first()
        if last is None:
            return []
        rows = []
        for opp in session.scalars(
            select(Opportunity).where(Opportunity.run_id == last.run_id).order_by(Opportunity.rank)
        ):
            cluster = session.get(Cluster, opp.cluster_id)
            sizing = latest_analysis(session, opp.cluster_id, AnalysisKind.SIZING)
            payload = sizing.payload if sizing else {}
            rows.append(
                {
                    "cluster_id": opp.cluster_id,
                    "rank": opp.rank,
                    "score": opp.score,
                    "name": cluster.name if cluster else "?",
                    "verdict": opp.verdict,
                    "factors": opp.breakdown["factors"],
                    "tam": payload.get("tam"),
                    "sam": payload.get("sam"),
                    "som": payload.get("som"),
                    "sourced_share": payload.get("sourced_share", 0.0),
                }
            )
        return rows
