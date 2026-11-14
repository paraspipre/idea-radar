"""`radar doctor`: environment, config, auth-safety and database checks. Never calls an LLM."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from idea_radar.config import ConfigError, Settings, load_settings
from idea_radar.db.migrate import current_revision, head_revision
from idea_radar.db.session import make_engine, redact_url
from idea_radar.llm.guard import api_credentials_in_env, remediation_message


class Status(StrEnum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info >= (3, 12):
        return Check("Python", Status.OK, version)
    return Check("Python", Status.FAIL, f"{version}; Idea Radar needs 3.12+")


def check_config(config_path: Path | None) -> tuple[Check, Settings | None]:
    try:
        settings = load_settings(config_path)
    except ConfigError as exc:
        return Check("Config", Status.FAIL, str(exc)), None
    detail = (
        f"backend={settings.llm.backend}, models small={settings.llm.models.small} "
        f"strong={settings.llm.models.strong}, "
        f"max calls/run={settings.budget.max_llm_calls_per_run}, focus='{settings.focus.name}'"
    )
    return Check("Config", Status.OK, detail), settings


def check_llm_auth(settings: Settings, environ: Mapping[str, str]) -> Check:
    found = api_credentials_in_env(environ)
    if settings.llm.backend != "agent_sdk":
        return Check(
            "LLM auth",
            Status.WARN,
            f"backend={settings.llm.backend} bills an API account, not your subscription.",
        )
    if not found:
        return Check(
            "LLM auth",
            Status.OK,
            "No API credentials in the environment; the Agent SDK will use your "
            "Claude subscription login.",
        )
    if settings.llm.allow_api_key_env:
        return Check(
            "LLM auth",
            Status.WARN,
            f"{', '.join(found)} set and llm.allow_api_key_env=true: calls may bill the API.",
        )
    return Check("LLM auth", Status.FAIL, remediation_message(found))


def check_data_dir(settings: Settings) -> Check:
    path = settings.data_path
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("Data dir", Status.FAIL, f"{path} is not writable: {exc}")
    if "onedrive" in str(path).lower():
        return Check(
            "Data dir",
            Status.WARN,
            f"{path} is inside OneDrive; cloud sync can lock or corrupt SQLite. "
            "Set RADAR__APP__DATA_DIR in .env to a folder outside OneDrive "
            "(see .env.example).",
        )
    return Check("Data dir", Status.OK, str(path))


def check_database(settings: Settings) -> Check:
    url = settings.db_url
    try:
        engine = make_engine(url)
        try:
            current = current_revision(engine)
        finally:
            engine.dispose()
        head = head_revision(url)
    except Exception as exc:  # doctor must report, never crash
        return Check("Database", Status.FAIL, f"{redact_url(url)}: {exc}")
    if current is None:
        return Check("Database", Status.WARN, "Not initialized. Run: uv run radar db init")
    if current != head:
        return Check(
            "Database",
            Status.WARN,
            f"At revision {current}, latest is {head}. Run: uv run radar db init",
        )
    return Check("Database", Status.OK, f"revision {current} at {redact_url(url)}")


def check_agent_sdk() -> Check:
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return Check("Agent SDK", Status.WARN, "claude-agent-sdk not installed yet (added in M1)")
    try:
        version = importlib.metadata.version("claude-agent-sdk")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown version"
    return Check("Agent SDK", Status.OK, f"claude-agent-sdk {version}")


def run_checks(config_path: Path | None, environ: Mapping[str, str]) -> list[Check]:
    checks = [check_python()]
    config_check, settings = check_config(config_path)
    checks.append(config_check)
    if settings is None:
        return checks
    # Config loading reads .env into os.environ, so check auth after it.
    checks += [
        check_llm_auth(settings, environ),
        check_data_dir(settings),
        check_database(settings),
        check_agent_sdk(),
    ]
    return checks
