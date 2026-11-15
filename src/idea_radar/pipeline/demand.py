"""Stage 4: demand score per active cluster (pure Python over the DB; no LLM)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.clock import utc_now
from idea_radar.config import Settings
from idea_radar.db.models import (
    Cluster,
    ClusterMember,
    ClusterStatus,
    DemandScore,
    Problem,
    RawItem,
)
from idea_radar.scoring.demand import (
    Signal,
    demand_scores,
    engagement_percentile,
    raw_components,
    weekly_counts,
)


@dataclass(frozen=True)
class RankedCluster:
    cluster_id: int
    name: str | None
    size: int
    score: float
    components: dict[str, dict[str, float]]


def run_demand(
    settings: Settings,
    factory: sessionmaker[Session],
    *,
    run_id: str | None,
    now: datetime | None = None,
) -> list[RankedCluster]:
    cfg = settings.demand
    now = now or utc_now()
    with factory() as session:
        engagement_by_source: dict[str, list[int]] = defaultdict(list)
        for source, score, comments in session.execute(
            select(RawItem.source, RawItem.score, RawItem.num_comments)
        ):
            engagement_by_source[source].append(int(score) + int(comments))
        for values in engagement_by_source.values():
            values.sort()

        clusters = {
            c.id: c
            for c in session.scalars(
                select(Cluster).where(Cluster.status == ClusterStatus.ACTIVE.value)
            )
        }
        signals: dict[int, list[Signal]] = defaultdict(list)
        seen: dict[int, set[str]] = defaultdict(set)  # one item counts once per cluster
        rows = session.execute(
            select(ClusterMember.cluster_id, Problem.wtp_strength, RawItem)
            .join(Problem, Problem.id == ClusterMember.problem_id)
            .join(RawItem, RawItem.id == Problem.raw_item_id)
        ).all()
        for cluster_id, wtp_strength, item in rows:
            if cluster_id not in clusters or item.id in seen[cluster_id]:
                continue
            seen[cluster_id].add(item.id)
            signals[cluster_id].append(
                Signal(
                    created_at=item.created_at,
                    source=item.source,
                    engagement_pct=engagement_percentile(
                        item.score + item.num_comments, engagement_by_source[item.source]
                    ),
                    wtp_strength=float(wtp_strength),
                )
            )
        raw = {
            cid: raw_components(
                signals[cid],
                now=now,
                half_life_days=cfg.half_life_days,
                window_days=cfg.trend_window_days,
            )
            for cid in clusters
            if signals[cid]
        }
        results = demand_scores(raw, cfg.weights)
        ranked = []
        for cid, result in results.items():
            components = {
                "raw": result.raw,
                "normalized": result.normalized,
                "weights": result.weights,
            }
            session.add(
                DemandScore(
                    cluster_id=cid,
                    run_id=run_id,
                    score=result.score,
                    components=components,
                    weekly_counts=weekly_counts([s.created_at for s in signals[cid]], now=now),
                )
            )
            ranked.append(
                RankedCluster(
                    cluster_id=cid,
                    name=clusters[cid].name,
                    size=len(signals[cid]),
                    score=result.score,
                    components=components,
                )
            )
        session.commit()
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
