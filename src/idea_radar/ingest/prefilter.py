"""Free prefilter: regex problem/WTP signals + engagement + source priors -> score.

Only items scoring at or above the threshold are ever sent to an LLM. Every decision keeps
its reasons, so prefilter recall can be measured against the hand-labeled eval set (M3).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from idea_radar.ingest.text import is_probably_english

_I = re.IGNORECASE


def _p(name: str, *parts: str) -> tuple[str, re.Pattern[str]]:
    """A named, case-insensitive pattern; ``parts`` are concatenated (keeps lines short)."""
    return name, re.compile("".join(parts), _I)


PROBLEM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    _p("wish", r"\bi wish (there (was|were|is)|i could|someone would)\b"),
    _p(
        "is_there_a",
        r"\bis there (a|an|any) (good |simple |easy |better )?",
        r"(tool|app|service|way|library|software|product|solution)\b",
    ),
    _p(
        "looking_for",
        r"\b(looking for|need|searching for) (a|an|some) (good |simple )?",
        r"(tool|app|service|solution|software|way|alternative)\b",
    ),
    _p(
        "pain_words",
        r"\b(frustrat\w*|annoy\w*|painful|tedious|nightmare|waste of time|time[- ]consuming|",
        r"struggl\w*|hate (it )?when|drives me crazy|so hard to)\b",
    ),
    _p(
        "how_do_you",
        r"\bhow do (you|people|teams|others) ",
        r"(deal with|handle|manage|keep track|track|cope)\b",
    ),
    _p("why_no", r"\bwhy (is there no|isn'?t there|can'?t i|doesn'?t \w+ (have|support))\b"),
    _p("alternative", r"\b(alternative to|replacement for|switch(ed|ing)? (from|away from))\b"),
    _p(
        "missing_feature",
        r"\b(doesn'?t|does not|can'?t|cannot|won'?t) ",
        r"(support|handle|let me|allow|sync|export)\b",
    ),
    _p("manual_work", r"\b(manually|by hand|copy[- ]past\w*|spreadsheet)\b"),
)

WTP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    _p("would_pay", r"\b(i'?d|i would|we'?d|we would) (happily |gladly |definitely )?pay\b"),
    _p("willing_to_pay", r"\b(willing|happy) to pay\b"),
    _p("take_my_money", r"\btake my money\b"),
    _p("price_point", r"\$\s?\d+(\.\d+)?\s?(/|per)\s?(mo|month|year|yr|user|seat)\b"),
    _p("too_expensive", r"\b(too expensive|overpriced|price (hike|increase))\b"),
    _p("paying_for", r"\b(currently |already )?paying (for|\$)"),
)

NEGATIVE_PATTERNS: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    ("hiring", 2.0, re.compile(r"\b(we'?re hiring|job opening|apply now|remote position)\b", _I)),
    ("self_promo", 1.0, re.compile(r"\b(check out my|i built|i made|we launched|show hn)\b", _I)),
    ("spam", 2.0, re.compile(r"\b(promo code|discount code|affiliate link)\b", _I)),
)

PROBLEM_WEIGHT = 1.0
PROBLEM_CAP = 2.0
WTP_WEIGHT = 1.5
WTP_CAP = 2.0
KEYWORD_WEIGHT = 0.25
KEYWORD_CAP = 0.5


@dataclass(frozen=True)
class PrefilterResult:
    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    first_signal_at: int | None = None  # where the snippet window should start

    @property
    def has_wtp(self) -> bool:
        return any(reason.startswith("wtp:") for reason in self.reasons)


def engagement_bonus(score: int, num_comments: int) -> float:
    """0..1, log-scaled: 100 combined points/comments saturates."""
    total = max(0, score) + max(0, num_comments)
    return min(1.0, math.log1p(total) / math.log(101))


def evaluate(
    text: str,
    *,
    source: str,
    score: int,
    num_comments: int,
    threshold: float,
    min_chars: int,
    source_prior: Mapping[str, float],
    focus_keywords: Sequence[str] = (),
) -> PrefilterResult:
    if len(text) < min_chars:
        return PrefilterResult(0.0, False, ["too_short"])
    if not is_probably_english(text):
        return PrefilterResult(0.0, False, ["not_english"])

    reasons: list[str] = []
    positions: list[int] = []

    problem = 0.0
    for name, pattern in PROBLEM_PATTERNS:
        match = pattern.search(text)
        if match:
            problem += PROBLEM_WEIGHT
            reasons.append(f"problem:{name}")
            positions.append(match.start())
    wtp = 0.0
    for name, pattern in WTP_PATTERNS:
        match = pattern.search(text)
        if match:
            wtp += WTP_WEIGHT
            reasons.append(f"wtp:{name}")
            positions.append(match.start())
    penalty = 0.0
    for name, weight, pattern in NEGATIVE_PATTERNS:
        if pattern.search(text):
            penalty += weight
            reasons.append(f"negative:{name}")

    lowered = text.lower()
    keyword_hits = sum(1 for keyword in focus_keywords if keyword.lower() in lowered)
    keyword_bonus = min(KEYWORD_CAP, keyword_hits * KEYWORD_WEIGHT)
    if keyword_hits:
        reasons.append(f"focus_keywords:{keyword_hits}")

    prior = source_prior.get(source, 0.0)
    if prior:
        reasons.append(f"source_prior:{prior:g}")
    engagement = engagement_bonus(score, num_comments)

    total = (
        min(problem, PROBLEM_CAP) + min(wtp, WTP_CAP) + keyword_bonus + prior + engagement - penalty
    )
    total = round(total, 3)
    return PrefilterResult(
        score=total,
        passed=total >= threshold,
        reasons=reasons,
        first_signal_at=min(positions) if positions else None,
    )
