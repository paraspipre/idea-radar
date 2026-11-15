"""Binary classification metrics for the prefilter and the extractor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from idea_radar.evals.gold import GoldItem
from idea_radar.pipeline.extract import ExtractedProblem


@dataclass(frozen=True)
class Binary:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[bool, bool]]) -> Binary:
        """pairs of (predicted, actual)."""
        tp = fp = fn = tn = 0
        for predicted, actual in pairs:
            if predicted and actual:
                tp += 1
            elif predicted:
                fp += 1
            elif actual:
                fn += 1
            else:
                tn += 1
        return cls(tp, fp, fn, tn)

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def prefilter_metrics(gold: Sequence[GoldItem]) -> Binary:
    """Did the free prefilter let the real problems through? (recall is what matters)"""
    return Binary.from_pairs((g.prefilter_status == "passed", g.is_problem) for g in gold)


def extraction_metrics(
    gold: Sequence[GoldItem], predictions: Mapping[str, ExtractedProblem | None]
) -> tuple[Binary, Binary]:
    """(is_problem metrics, willingness-to-pay metrics) over gold items that were predicted."""
    scored = [g for g in gold if g.id in predictions]
    problem = Binary.from_pairs((predictions[g.id] is not None, g.is_problem) for g in scored)
    wtp = Binary.from_pairs(
        (
            (p := predictions[g.id]) is not None and p.wtp == "strong",
            g.is_problem and g.wtp,
        )
        for g in scored
    )
    return problem, wtp


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
