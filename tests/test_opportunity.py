"""Stages 5-8: sizing validation math, scores, and the stages end to end with FakeBackend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.config import Settings, load_settings
from idea_radar.db.models import Analysis, Cluster, DemandScore, MarketFact, Opportunity
from idea_radar.embed import HashEmbedder
from idea_radar.llm.factory import build_client
from idea_radar.llm.fake import FakeBackend
from idea_radar.pipeline.graph import PipelineContext, run_pipeline
from idea_radar.pipeline.opportunity import is_stale
from idea_radar.scoring.opportunity import (
    SizingInput,
    competition_score,
    compute_sizing,
    feasibility_score,
    monetization_score,
    opportunity_score,
)
from tests.test_cluster_demand import CORPUS, seed


def inp(low: float, base: float, high: float, **kw: Any) -> SizingInput:
    fields: dict[str, Any] = {"basis": "assumption", "assumption": "x", "confidence": "medium"}
    fields.update(kw)
    return SizingInput(low=low, base=base, high=high, **fields)


def test_sizing_math_and_validation() -> None:
    totals = compute_sizing(
        {
            "population": inp(1000, 2000, 4000, basis="fact", fact_id="F1", confidence="high"),
            "share_with_problem": inp(0.5, 0.2, 0.1),  # reversed range -> reordered
            "price_per_year": inp(100, 200, 300, basis="fact", fact_id="F99"),  # unknown fact
            "reachable_share": inp(0.1, 0.5, 2.0),  # share > 1 -> clamped
            "obtainable_share": inp(0.1, 0.1, 0.1, assumption=None),  # no rationale
        },
        {"F1"},
    )
    assert totals.tam == (1000 * 0.1 * 100, 2000 * 0.2 * 200, 4000 * 0.5 * 300)
    assert totals.sam[2] == totals.tam[2] * 1.0
    assert totals.som[1] == pytest.approx(totals.sam[1] * 0.1)
    assert totals.inputs["population"].basis == "fact"
    price = totals.inputs["price_per_year"]
    assert price.basis == "assumption" and price.confidence == "low" and price.issues
    assert "range reordered" in totals.inputs["share_with_problem"].issues
    assert "share clamped to 1" in totals.inputs["reachable_share"].issues
    assert totals.inputs["obtainable_share"].confidence == "low"
    assert totals.sourced_share == 0.2


def test_scores() -> None:
    assert competition_score(1.0, 1.0) == 0.5  # crowded but hated -> half as closed
    assert competition_score(0.2, 0.0) == 0.2
    assert monetization_score(0, 0) == 0.0
    assert monetization_score(1e8, 1.0) == 1.0
    from idea_radar.scoring.opportunity import Feasibility

    easy = Feasibility(build_complexity=5, regulation=5, data_access=5, distribution=5, notes="")
    assert feasibility_score(easy) == 1.0
    s = opportunity_score(demand=0.8, competition=1.0, monetization=0.5, feasibility=0.5)
    assert s.raw_product == 0.0 and s.score > 0  # floored, but the raw product is kept


COMPETITION = {
    "competitors": [{"name": "Acme", "url": "https://acme.test", "positioning": "x"}],
    "crowdedness": 0.4,
    "dissatisfaction": 0.8,
    "rationale": "r",
    "facts": [
        {
            "claim": "There are 50,000 freelance bookkeepers",
            "value": 50000,
            "unit": "people",
            "source_url": "https://stats.test/a",
            "quote": "50,000 freelance bookkeepers",
        },
        {"claim": "bad", "source_url": "not-a-url", "quote": "q"},
    ],
}


def sizing_reply(fact_ref: str) -> dict[str, Any]:
    def i(low: float, base: float, high: float, fact: str | None = None) -> dict[str, Any]:
        if fact:
            return {"low": low, "base": base, "high": high, "basis": "fact", "fact_id": fact,
                    "confidence": "high"}  # fmt: skip
        return {"low": low, "base": base, "high": high, "basis": "assumption",
                "assumption": "guess", "confidence": "medium"}  # fmt: skip

    return {
        "population": i(40000, 50000, 60000, fact_ref),
        "share_with_problem": i(0.2, 0.3, 0.4),
        "price_per_year": i(100, 200, 300),
        "reachable_share": i(0.05, 0.1, 0.2),
        "obtainable_share": i(0.05, 0.1, 0.2),
        "feasibility": {
            "build_complexity": 4,
            "regulation": 4,
            "data_access": 3,
            "distribution": 3,
            "notes": "n",
        },
        "rationale": "r",
    }


SKEPTIC = {
    "verdict": "weak",
    "objections": [{"claim": "Mostly one forum", "severity": "high"}],
    "what_would_change_my_mind": "Evidence from paying users",
    "confidence": "medium",
}


@pytest.fixture
def settings(config_path: Path) -> Settings:
    s = load_settings(config_path, environ={}, load_env_file=False)
    clustering = s.clustering.model_copy(
        update={
            "assign_threshold": 0.95,
            "merge_threshold": 0.97,
            "min_pool": 4,
            "min_member_similarity": 0.8,
        }
    )
    demand = s.demand.model_copy(update={"top_n": 1})
    return s.model_copy(update={"clustering": clustering, "demand": demand})


def test_stages_5_to_8_end_to_end(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    seed(session_factory, CORPUS)
    naming = {
        "clusters": [
            {"k": 0, "name": "A", "summary": "S", "who": "W"},
            {"k": 1, "name": "B", "summary": "S", "who": "W"},
        ]
    }
    backend = FakeBackend([naming, COMPETITION])
    client = build_client(settings, factory=session_factory, run_id=None, backend=backend)

    async def go() -> PipelineContext:
        ctx = PipelineContext(
            settings=settings,
            factory=session_factory,
            client=client,
            embedder=HashEmbedder(),
            run_id="r1",
            skip=frozenset({"extract"}),
        )
        ctx.extra_stages = []
        from idea_radar.pipeline.graph import EXTRA_STAGES

        ctx.extra_stages = list(EXTRA_STAGES)
        # sizing must cite the fact id the competition stage stored; queue it lazily
        original = backend.complete

        async def complete(request: Any, **kw: Any) -> Any:
            if request.stage == "sizing":
                fact_id = request.prompt.split("[F", 1)[1].split("]", 1)[0]
                backend.queue(sizing_reply(f"F{fact_id}"), SKEPTIC)
            return await original(request, **kw)

        backend.complete = complete  # type: ignore[method-assign]
        await run_pipeline(ctx)
        return ctx

    from idea_radar.db.models import Run

    with session_factory() as session:
        session.add(Run(id="r1", kind="pipeline", config_hash="x"))
        session.commit()
    ctx = asyncio.run(go())
    assert ctx.stop is None
    stages = [c.request.stage for c in backend.calls]
    assert stages == ["name_clusters", "competition", "sizing", "skeptic"]
    competition_call = backend.calls[1].request
    assert competition_call.tools == ("WebSearch", "WebFetch") and competition_call.tier == "strong"

    with session_factory() as session:
        facts = list(session.scalars(select(MarketFact)))
        analyses = {a.kind: a for a in session.scalars(select(Analysis))}
        opportunities = list(session.scalars(select(Opportunity)))
        cluster = session.get(Cluster, opportunities[0].cluster_id)
        assert cluster is not None and cluster.name in {"A", "B"}
        assert session.scalars(select(DemandScore)).first() is not None
    assert len(facts) == 1  # the fact without a URL was dropped
    sizing = analyses["sizing"].payload
    assert sizing["inputs"]["population"]["basis"] == "fact"
    assert sizing["inputs"]["population"]["source_url"] == "https://stats.test/a"
    assert sizing["tam"][1] == pytest.approx(50000 * 0.3 * 200)
    assert sizing["sourced_share"] == 0.2
    assert len(opportunities) == 1 and opportunities[0].rank == 1
    assert opportunities[0].verdict == "weak"

    # second run: everything fresh -> no new LLM calls for stages 5-7
    backend.calls.clear()
    with session_factory() as session:
        session.add(Run(id="r2", kind="pipeline", config_hash="x"))
        session.commit()

    async def again() -> None:
        ctx2 = PipelineContext(
            settings=settings,
            factory=session_factory,
            client=client,
            embedder=HashEmbedder(),
            run_id="r2",
            skip=frozenset({"extract"}),
        )
        from idea_radar.pipeline.graph import EXTRA_STAGES

        ctx2.extra_stages = list(EXTRA_STAGES)
        await run_pipeline(ctx2)

    asyncio.run(again())
    assert backend.calls == []


def test_staleness_rules() -> None:
    a = Analysis(cluster_id=1, kind="sizing", cluster_size=4, payload={})
    from idea_radar.clock import utc_now

    a.created_at = utc_now()
    a.id = 5
    assert not is_stale(a, 5)  # +25% exactly is not "grown"
    assert is_stale(a, 6)
    assert is_stale(None, 1)
    newer = Analysis(cluster_id=1, kind="competition", cluster_size=4, payload={})
    newer.created_at = a.created_at
    newer.id = 9
    assert is_stale(a, 4, after=newer)
