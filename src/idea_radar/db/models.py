"""ORM models. Tables are added milestone by milestone, each with its own migration."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import Integer, String, Text
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
