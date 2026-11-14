"""`radar llm smoke` and `radar budget` end to end, with FakeBackend instead of a real model."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from idea_radar import cli, exit_codes
from idea_radar.config import Settings
from idea_radar.llm.errors import RateLimited
from idea_radar.llm.fake import FakeBackend, FakeReply

runner = CliRunner()


@pytest.fixture
def initialized(config_path: Path) -> Path:
    result = runner.invoke(cli.app, ["--config", str(config_path), "db", "init"])
    assert result.exit_code == 0, result.output
    return config_path


def use_fake(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> None:
    def fake_build_backend(settings: Settings) -> FakeBackend:
        return backend

    monkeypatch.setattr(cli, "build_backend", fake_build_backend)


def test_smoke_success_then_budget_shows_it(
    initialized: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch, FakeBackend([{"ok": True, "echo": "radar"}]))
    result = runner.invoke(cli.app, ["--config", str(initialized), "llm", "smoke"])
    assert result.exit_code == 0, result.output
    assert "echo='radar'" in result.output
    assert "Input tokens:   100" in result.output

    budget = runner.invoke(cli.app, ["--config", str(initialized), "budget"])
    assert budget.exit_code == 0, budget.output
    assert "smoke" in budget.output
    assert "completed" in budget.output


def test_smoke_rate_limited_exits_75_and_blocks_next_run(
    initialized: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resets = datetime(2099, 1, 1, tzinfo=UTC)
    use_fake(monkeypatch, FakeBackend([FakeReply(error=RateLimited("limit", resets_at=resets))]))
    first = runner.invoke(cli.app, ["--config", str(initialized), "llm", "smoke"])
    assert first.exit_code == exit_codes.TEMPFAIL, first.output

    backend = FakeBackend([{"ok": True, "echo": "radar"}])
    use_fake(monkeypatch, backend)
    second = runner.invoke(cli.app, ["--config", str(initialized), "llm", "smoke"])
    assert second.exit_code == exit_codes.TEMPFAIL
    assert "Usage limit active until 2099-01-01" in second.output
    assert backend.calls == []  # never called while blocked


def test_smoke_refuses_when_api_key_is_set(
    initialized: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    result = runner.invoke(cli.app, ["--config", str(initialized), "llm", "smoke"])
    assert result.exit_code == exit_codes.FAILURE
    assert "ANTHROPIC_API_KEY" in result.output


def test_commands_require_migrated_db(config_path: Path) -> None:
    result = runner.invoke(cli.app, ["--config", str(config_path), "budget"])
    assert result.exit_code == exit_codes.FAILURE
    assert "radar db init" in result.output
