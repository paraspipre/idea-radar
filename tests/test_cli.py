from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from idea_radar.cli import app
from idea_radar.db.migrate import head_revision

runner = CliRunner()
HEAD = head_revision("sqlite://")


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "config", "db", "version"):
        assert command in result.output


def test_doctor_passes_on_fresh_project(config_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(config_path), "doctor"])
    assert result.exit_code == 0, result.output
    assert "Not initialized" in result.output


def test_db_init_then_status_then_doctor(config_path: Path) -> None:
    init = runner.invoke(app, ["--config", str(config_path), "db", "init"])
    assert init.exit_code == 0, init.output
    assert f"revision {HEAD}" in init.output

    status = runner.invoke(app, ["--config", str(config_path), "db", "status"])
    assert status.exit_code == 0, status.output
    assert "Runs:     0" in status.output

    doctor = runner.invoke(app, ["--config", str(config_path), "doctor"])
    assert doctor.exit_code == 0, doctor.output
    assert f"revision {HEAD}" in doctor.output


def test_db_status_before_init_exits_nonzero(config_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(config_path), "db", "status"])
    assert result.exit_code == 1
    assert "not initialized" in result.output


def test_doctor_fails_when_api_key_is_set(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    result = runner.invoke(app, ["--config", str(config_path), "doctor"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_doctor_catches_api_key_in_dotenv(config_path: Path, project: Path) -> None:
    (project / ".env").write_text("ANTHROPIC_API_KEY=sk-from-dotenv\n", encoding="utf-8")
    try:
        result = runner.invoke(app, ["--config", str(config_path), "doctor"])
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output
    finally:
        # load_dotenv wrote into os.environ; the autouse fixture only cleans what existed before.
        os.environ.pop("ANTHROPIC_API_KEY", None)


def test_config_command_prints_effective_config(config_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--config", str(config_path), "config"],
        env={"RADAR__LLM__MODELS__SMALL": "sonnet"},
    )
    assert result.exit_code == 0, result.output
    assert "small: sonnet" in result.output
    assert "fingerprint" in result.output


def test_bad_config_exits_with_code_2(project: Path) -> None:
    bad = project / "config" / "default.yaml"
    bad.write_text("llm:\n  backend: nonsense\n", encoding="utf-8")
    result = runner.invoke(app, ["--config", str(bad), "config"])
    assert result.exit_code == 2
