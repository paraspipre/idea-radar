from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

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
