"""Stages 6 and 8 math (pure): bottom-up sizing from sourced inputs, and the opportunity score.

Sizing: the LLM only proposes *inputs* (population, shares, price), each with low/base/high
and a basis: a stored market fact (by id) or an explicit assumption. Python validates the
basis and does every multiplication, so totals can't be invented and arithmetic can't drift.

    TAM = population x share_with_problem x price_per_year
    SAM = TAM x reachable_share
    SOM = SAM x obtainable_share

Opportunity = demand x (1 - competition) x monetization x feasibility, each floored at 0.05
so one noisy zero can't hide a result (the unfloored product is kept too).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

Basis = Literal["fact", "assumption"]
Confidence = Literal["low", "medium", "high"]
SHARE_KEYS = ("share_with_problem", "reachable_share", "obtainable_share")
INPUT_KEYS = ("population", "share_with_problem", "price_per_year", *SHARE_KEYS[1:])
FLOOR = 0.05


class SizingInput(BaseModel):
    low: float = Field(ge=0)
    base: float = Field(ge=0)
    high: float = Field(ge=0)
    basis: Basis
    fact_id: str | None = None
    assumption: str | None = None
    confidence: Confidence


class Feasibility(BaseModel):
    """1 = very hard, 5 = easy, for a small team."""

    build_complexity: int = Field(ge=1, le=5)
    regulation: int = Field(ge=1, le=5)
    data_access: int = Field(ge=1, le=5)
    distribution: int = Field(ge=1, le=5)
    notes: str


@dataclass(frozen=True)
class CheckedInput:
    key: str
    low: float
    base: float
    high: float
    basis: Basis
    fact_id: str | None
    assumption: str | None
    confidence: Confidence
    issues: tuple[str, ...] = ()


def check_input(key: str, item: SizingInput, fact_ids: set[str]) -> CheckedInput:
    """Enforce: sourced or explicitly assumed; ordered range; shares within [0, 1]."""
    issues: list[str] = []
    low, base, high = sorted((item.low, item.base, item.high))
    if (low, base, high) != (item.low, item.base, item.high):
        issues.append("range reordered")
    if key in SHARE_KEYS and high > 1:
        issues.append("share clamped to 1")
        low, base, high = min(low, 1.0), min(base, 1.0), min(high, 1.0)
    basis, confidence = item.basis, item.confidence
    assumption = (item.assumption or "").strip() or None
    fact_id = item.fact_id
    if basis == "fact" and fact_id not in fact_ids:
        issues.append(f"cited fact {fact_id!r} does not exist; treated as an assumption")
        basis, confidence, fact_id = "assumption", "low", None
        assumption = assumption or "Unsourced number proposed by the model."
    if basis == "assumption" and assumption is None:
        issues.append("assumption had no stated rationale")
        assumption, confidence = "No rationale given.", "low"
    return CheckedInput(key, low, base, high, basis, fact_id, assumption, confidence, tuple(issues))


@dataclass(frozen=True)
class SizingTotals:
    tam: tuple[float, float, float]
    sam: tuple[float, float, float]
    som: tuple[float, float, float]
    inputs: dict[str, CheckedInput] = field(default_factory=dict)

    @property
    def sourced_share(self) -> float:
        """Fraction of inputs backed by a stored fact (shown as a confidence signal)."""
        return sum(1 for i in self.inputs.values() if i.basis == "fact") / len(self.inputs)


def compute_sizing(inputs: Mapping[str, SizingInput], fact_ids: set[str]) -> SizingTotals:
    checked = {key: check_input(key, inputs[key], fact_ids) for key in INPUT_KEYS}

    def triple(*keys: str) -> tuple[float, float, float]:
        lows = math.prod(checked[k].low for k in keys)
        bases = math.prod(checked[k].base for k in keys)
        highs = math.prod(checked[k].high for k in keys)
        return (lows, bases, highs)

    tam_keys = ("population", "share_with_problem", "price_per_year")
    return SizingTotals(
        tam=triple(*tam_keys),
        sam=triple(*tam_keys, "reachable_share"),
        som=triple(*tam_keys, "reachable_share", "obtainable_share"),
        inputs=checked,
    )


def monetization_score(som_base: float, wtp: float) -> float:
    """0..1: log-scaled obtainable revenue ($100M/yr SOM saturates) blended with WTP signals."""
    scale = min(1.0, math.log10(som_base + 1) / 8) if som_base > 0 else 0.0
    return round(0.7 * scale + 0.3 * max(0.0, min(1.0, wtp)), 4)


def feasibility_score(f: Feasibility) -> float:
    parts = (f.build_complexity, f.regulation, f.data_access, f.distribution)
    return round(sum((p - 1) / 4 for p in parts) / len(parts), 4)


def competition_score(crowdedness: float, dissatisfaction: float) -> float:
    """0..1, higher = harder. A crowded space whose users hate the incumbents is less closed."""
    c = max(0.0, min(1.0, crowdedness))
    d = max(0.0, min(1.0, dissatisfaction))
    return round(c * (1 - 0.5 * d), 4)


@dataclass(frozen=True)
class OpportunityScore:
    score: float
    raw_product: float
    factors: dict[str, float]


def opportunity_score(
    *, demand: float, competition: float, monetization: float, feasibility: float
) -> OpportunityScore:
    factors = {
        "demand": demand,
        "1-competition": 1 - competition,
        "monetization": monetization,
        "feasibility": feasibility,
    }
    raw = math.prod(factors.values())
    floored = math.prod(max(FLOOR, v) for v in factors.values())
    return OpportunityScore(
        score=round(floored, 5),
        raw_product=round(raw, 5),
        factors={k: round(v, 4) for k, v in factors.items()},
    )
