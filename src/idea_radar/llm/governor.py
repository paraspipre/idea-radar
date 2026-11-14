"""Budget governor: caps LLM calls per run and per stage, and stops early on rate-limit signals.

Cache hits are free and never touch the governor. Every real attempt (including retries)
counts, because every attempt can consume subscription usage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from idea_radar.llm.errors import BudgetExhausted
from idea_radar.llm.types import RateLimitStatus


@dataclass
class BudgetGovernor:
    max_calls: int
    per_stage: Mapping[str, int] = field(default_factory=dict)
    stop_on_warning: bool = True

    calls: int = 0
    stage_calls: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    resets_at: datetime | None = None

    def remaining(self, stage: str) -> int:
        total_left = self.max_calls - self.calls
        if stage in self.per_stage:
            stage_left = self.per_stage[stage] - self.stage_calls.get(stage, 0)
            return max(0, min(total_left, stage_left))
        return max(0, total_left)

    def ensure_can_call(self, stage: str) -> None:
        if self.stop_reason is not None:
            raise BudgetExhausted(self.stop_reason)
        if self.calls >= self.max_calls:
            raise BudgetExhausted(f"Run budget used up ({self.calls}/{self.max_calls} calls).")
        if stage in self.per_stage and self.stage_calls.get(stage, 0) >= self.per_stage[stage]:
            raise BudgetExhausted(
                f"Stage budget for {stage!r} used up ({self.per_stage[stage]} calls)."
            )

    def record_call(self, stage: str) -> None:
        self.calls += 1
        self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1

    def observe_rate_limit(self, status: RateLimitStatus, resets_at: datetime | None) -> None:
        if resets_at is not None:
            self.resets_at = resets_at
        if status == "rejected":
            self.stop_reason = "Claude usage limit reached; resuming after it resets."
        elif status == "allowed_warning" and self.stop_on_warning:
            self.stop_reason = (
                "Approaching the Claude usage limit; stopping early to leave headroom."
            )
