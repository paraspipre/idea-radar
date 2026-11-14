"""ORM models. Tables are added milestone by milestone, each with its own migration."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from idea_radar.clock import utc_now
from idea_radar.db.base import Base
from idea_radar.db.types import UTCDateTime


def new_id() -> str:
    return uuid.uuid4().hex


class RunKind(StrEnum):
    INGEST = "ingest"
    PIPELINE = "pipeline"
    EVAL = "eval"
    REPORT = "report"
    SMOKE = "smoke"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


class Run(Base):
    """One invocation of ingest / pipeline / eval / report. Feeds the budget panel."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default=RunStatus.RUNNING.value)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True, default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    calls_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    config_hash: Mapped[str] = mapped_column(String(64))
    # When a rate limit stopped the run: the next run won't start before this time.
    resets_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    notes: Mapped[str | None] = mapped_column(Text)


class LLMCallStatus(StrEnum):
    OK = "ok"
    CACHE_HIT = "cache_hit"
    RATE_LIMITED = "rate_limited"
    TRANSIENT_ERROR = "transient_error"
    OUTPUT_ERROR = "output_error"
    ERROR = "error"


class LLMCacheEntry(Base):
    """Validated LLM outputs keyed by a hash of everything that determines them."""

    __tablename__ = "llm_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    stage: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(64))
    prompt_name: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[int] = mapped_column(Integer)
    response: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class LLMCall(Base):
    """One attempted LLM call (or cache hit). Feeds `radar budget` and the dashboard panel."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), index=True)
    stage: Mapped[str] = mapped_column(String(64))
    prompt: Mapped[str] = mapped_column(String(80))  # e.g. "extract.v1"
    backend: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    attempt: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    n_items: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cache_creation_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost_usd_est: Mapped[float | None] = mapped_column(Float)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    auth_source: Mapped[str | None] = mapped_column(String(64))
    rate_limit_status: Mapped[str | None] = mapped_column(String(32))
    rate_limit_utilization: Mapped[float | None] = mapped_column(Float)
    cache_key: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True, default=utc_now)
