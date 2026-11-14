from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.db.migrate import upgrade_to_head
from idea_radar.db.session import make_engine, make_session_factory

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from the developer's real environment."""
    for name in list(os.environ):
        if name.upper().startswith("RADAR__") or name.upper() in {
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
        }:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A throwaway project root containing a copy of the real default config."""
    (tmp_path / "config").mkdir()
    shutil.copy(REPO_ROOT / "config" / "default.yaml", tmp_path / "config" / "default.yaml")
    return tmp_path


@pytest.fixture
def config_path(project: Path) -> Path:
    return project / "config" / "default.yaml"


@pytest.fixture
def session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    """A migrated throwaway SQLite database."""
    url = f"sqlite:///{(tmp_path / 'db' / 'test.db').as_posix()}"
    upgrade_to_head(url)
    engine = make_engine(url)
    yield make_session_factory(engine)
    engine.dispose()
