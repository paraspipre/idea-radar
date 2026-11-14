"""Run bookkeeping: start/finish runs and honour rate-limit reset times across runs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.clock import utc_now
from idea_radar.db.models import Run, RunStatus


def start_run(factory: sessionmaker[Session], *, kind: str, config_hash: str) -> str:
    with factory() as session:
        run = Run(kind=kind, config_hash=config_hash, status=RunStatus.RUNNING.value)
        session.add(run)
        session.commit()
        return run.id


def finish_run(
    factory: sessionmaker[Session],
    run_id: str,
    *,
    status: RunStatus,
    calls_used: int,
    resets_at: datetime | None = None,
    notes: str | None = None,
) -> None:
    with factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise KeyError(f"Unknown run {run_id}")
        run.status = status.value
        run.finished_at = utc_now()
        run.calls_used = calls_used
        run.resets_at = resets_at
        run.notes = notes
        session.commit()


def rate_limit_block(factory: sessionmaker[Session], *, now: datetime) -> datetime | None:
    """If an earlier run hit the usage limit and it hasn't reset yet, return the reset time."""
    stmt = (
        select(Run.resets_at)
        .where(Run.status == RunStatus.RATE_LIMITED.value, Run.resets_at.is_not(None))
        .order_by(Run.resets_at.desc())
        .limit(1)
    )
    with factory() as session:
        resets_at = session.scalar(stmt)
    if resets_at is not None and resets_at > now:
        return resets_at
    return None
