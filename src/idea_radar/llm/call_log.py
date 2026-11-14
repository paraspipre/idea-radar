"""Persist one row per LLM attempt or cache hit (the source of the budget panel)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from idea_radar.db.models import LLMCall, LLMCallStatus
from idea_radar.llm.types import LLMUsage, RateLimitSnapshot


@dataclass(frozen=True)
class CallRecord:
    run_id: str | None
    stage: str
    prompt: str
    backend: str
    model: str
    status: LLMCallStatus
    attempt: int
    n_items: int
    usage: LLMUsage
    cache_key: str | None = None
    auth_source: str | None = None
    rate_limit: RateLimitSnapshot | None = None
    error: str | None = None


class CallLog:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def record(self, record: CallRecord) -> None:
        usage = record.usage
        with self._factory() as session:
            session.add(
                LLMCall(
                    run_id=record.run_id,
                    stage=record.stage,
                    prompt=record.prompt,
                    backend=record.backend,
                    model=record.model,
                    status=record.status.value,
                    attempt=record.attempt,
                    n_items=record.n_items,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_creation_tokens=usage.cache_creation_tokens,
                    cost_usd_est=usage.cost_usd_est,
                    duration_ms=usage.duration_ms,
                    auth_source=record.auth_source,
                    rate_limit_status=record.rate_limit.status if record.rate_limit else None,
                    rate_limit_utilization=(
                        record.rate_limit.utilization if record.rate_limit else None
                    ),
                    cache_key=record.cache_key,
                    error=(record.error or "")[:2000] or None,
                )
            )
            session.commit()
