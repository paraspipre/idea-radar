"""Demand score (stage 4): transparent, weighted, every component kept for the breakdown.

Components, each in [0, 1] before weighting:
- volume:        distinct items in the cluster (log-scaled, percentile-ranked across clusters)
- trend:         last window vs the window before it (log ratio, percentile-ranked)
- recency:       mean exponential decay of item age (half-life in days)
- engagement:    mean per-source engagement percentile of the items (HN points != GitHub reactions)
- wtp:           mean willingness-to-pay strength (none 0, weak 0.5, strong 1)
- corroboration: distinct sources (1 -> 0, 2 -> 0.6, 3+ -> 1): one forum's echo chamber is weak
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

RELATIVE = ("volume", "trend")  # unbounded: ranked across clusters
ABSOLUTE = ("recency", "engagement", "wtp", "corroboration")  # already in [0, 1]


@dataclass(frozen=True)
class Signal:
    """One item behind a cluster."""

    created_at: datetime
    source: str
    engagement_pct: float
    wtp_strength: float


def corroboration(n_sources: int) -> float:
    return 0.0 if n_sources <= 1 else 0.6 if n_sources == 2 else 1.0


def raw_components(
    signals: Sequence[Signal], *, now: datetime, half_life_days: float, window_days: int
) -> dict[str, float]:
    if not signals:
        return dict.fromkeys(RELATIVE + ABSOLUTE, 0.0)
    ages = [max(0.0, (now - s.created_at).total_seconds() / 86400) for s in signals]
    recent = sum(1 for a in ages if a <= window_days)
    prior = sum(1 for a in ages if window_days < a <= 2 * window_days)
    return {
        "volume": math.log1p(len(signals)),
        "trend": math.log((recent + 1) / (prior + 1)),
        "recency": sum(0.5 ** (a / half_life_days) for a in ages) / len(ages),
        "engagement": sum(s.engagement_pct for s in signals) / len(signals),
        "wtp": sum(s.wtp_strength for s in signals) / len(signals),
        "corroboration": corroboration(len({s.source for s in signals})),
    }


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Average-rank percentile in [0, 1]; ties share a rank; a single value maps to 0.5."""
    if len(values) == 1:
        return [0.5]
    ordered = sorted(values)
    n = len(values)
    out = []
    for v in values:
        lo = bisect.bisect_left(ordered, v)
        hi = bisect.bisect_right(ordered, v) - 1
        out.append(((lo + hi) / 2) / (n - 1))
    return out


@dataclass(frozen=True)
class DemandResult:
    score: float
    raw: dict[str, float]
    normalized: dict[str, float]
    weights: dict[str, float]


def demand_scores(
    raw_by_cluster: Mapping[int, dict[str, float]], weights: Mapping[str, float]
) -> dict[int, DemandResult]:
    ids = list(raw_by_cluster)
    if not ids:
        return {}
    normalized: dict[int, dict[str, float]] = {cid: {} for cid in ids}
    for key in RELATIVE:
        ranks = percentile_ranks([raw_by_cluster[cid][key] for cid in ids])
        for cid, rank in zip(ids, ranks, strict=True):
            normalized[cid][key] = rank
    for cid in ids:
        for key in ABSOLUTE:
            normalized[cid][key] = min(1.0, max(0.0, raw_by_cluster[cid][key]))
    total_weight = sum(weights.values()) or 1.0
    results = {}
    for cid in ids:
        score = sum(w * normalized[cid].get(k, 0.0) for k, w in weights.items()) / total_weight
        results[cid] = DemandResult(
            score=round(score, 4),
            raw={k: round(v, 4) for k, v in raw_by_cluster[cid].items()},
            normalized={k: round(v, 4) for k, v in normalized[cid].items()},
            weights=dict(weights),
        )
    return results


def weekly_counts(dates: Sequence[datetime], *, now: datetime, weeks: int = 8) -> list[int]:
    """Items per week, oldest week first, for the trend chart."""
    counts = [0] * weeks
    for d in dates:
        index = int((now - d) / timedelta(weeks=1))
        if 0 <= index < weeks:
            counts[weeks - 1 - index] += 1
    return counts


def engagement_percentile(value: int, sorted_source_values: Sequence[int]) -> float:
    """Where ``value`` sits among all items from the same source (0..1)."""
    if not sorted_source_values:
        return 0.0
    position = bisect.bisect_right(sorted_source_values, value)
    return position / len(sorted_source_values)
