from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine
from sqlalchemy.exc import StatementError

from idea_radar.db.base import Base
from idea_radar.db.migrate import current_revision, head_revision, upgrade_to_head
from idea_radar.db.models import Run, RunKind, RunStatus
from idea_radar.db.session import make_engine, make_session_factory


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'nested' / 'test.db').as_posix()}"


@pytest.fixture
def migrated_engine(db_url: str) -> Iterator[Engine]:
    upgrade_to_head(db_url)
    engine = make_engine(db_url)
    yield engine
    engine.dispose()


def test_upgrade_reaches_head(migrated_engine: Engine, db_url: str) -> None:
    head = head_revision(db_url)
    assert head is not None
    assert current_revision(migrated_engine) == head


def test_upgrade_is_idempotent(migrated_engine: Engine, db_url: str) -> None:
    upgrade_to_head(db_url)
    assert current_revision(migrated_engine) == head_revision(db_url)


def test_models_match_migrations(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": False})
        diff = compare_metadata(context, Base.metadata)
    assert diff == [], f"Models and migrations differ; add a migration: {diff}"


def test_run_roundtrip_is_timezone_aware(migrated_engine: Engine) -> None:
    factory = make_session_factory(migrated_engine)
    with factory() as session:
        run = Run(kind=RunKind.PIPELINE.value, config_hash="x" * 64)
        session.add(run)
        session.commit()
        run_id = run.id

    with factory() as session:
        loaded = session.get(Run, run_id)
        assert loaded is not None
        assert loaded.status == RunStatus.RUNNING.value
        assert loaded.calls_used == 0
        assert loaded.started_at.tzinfo is not None
        assert loaded.started_at.utcoffset() == timedelta(0)


def test_naive_datetime_is_rejected(migrated_engine: Engine) -> None:
    factory = make_session_factory(migrated_engine)
    with factory() as session:
        session.add(Run(kind="eval", config_hash="y", started_at=datetime(2026, 1, 1)))
        with pytest.raises(StatementError, match="timezone-aware"):
            session.commit()
