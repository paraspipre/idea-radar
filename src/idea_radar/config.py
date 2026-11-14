"""Configuration loading.

Order (later wins): ``config/default.yaml`` -> ``config/local.yaml`` -> ``RADAR__*`` env vars.
A ``.env`` file in the project root is loaded into the environment first (never overriding
variables that are already set).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    model_validator,
)

ENV_PREFIX = "RADAR__"
CONFIG_DIR_NAME = "config"
DEFAULT_CONFIG_NAME = "default.yaml"
LOCAL_CONFIG_NAME = "local.yaml"


class ConfigError(Exception):
    """Raised when configuration cannot be found, parsed, or validated."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AppConfig(_Section):
    data_dir: Path = Path("data")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"


class DbConfig(_Section):
    url: str | None = None


class LLMModels(_Section):
    small: str = "haiku"
    strong: str = "sonnet"


class LLMConfig(_Section):
    backend: Literal["agent_sdk", "anthropic_api"] = "agent_sdk"
    models: LLMModels = Field(default_factory=LLMModels)
    allow_api_key_env: bool = False


class BudgetConfig(_Section):
    max_llm_calls_per_run: PositiveInt = 21
    per_stage: dict[str, NonNegativeInt] = Field(default_factory=dict)
    stop_on_rate_limit_warning: bool = True

    @model_validator(mode="after")
    def _stage_budgets_fit(self) -> Self:
        for stage, limit in self.per_stage.items():
            if limit > self.max_llm_calls_per_run:
                raise ValueError(
                    f"budget.per_stage.{stage}={limit} exceeds "
                    f"budget.max_llm_calls_per_run={self.max_llm_calls_per_run}"
                )
        return self


class FocusConfig(_Section):
    name: str = "General"
    description: str = ""
    keywords: list[str] = Field(default_factory=list)


class Settings(_Section):
    app: AppConfig = Field(default_factory=AppConfig)
    db: DbConfig = Field(default_factory=DbConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    focus: FocusConfig = Field(default_factory=FocusConfig)
    # Project root; set by the loader, never read from YAML dumps.
    root: Path = Field(default=Path("."), exclude=True)

    @property
    def data_path(self) -> Path:
        path = self.app.data_dir
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def db_url(self) -> str:
        if self.db.url:
            return self.db.url
        return f"sqlite:///{(self.data_path / 'radar.db').as_posix()}"

    def fingerprint(self) -> str:
        """Stable hash of the effective config (stored on each run for reproducibility)."""
        payload = self.model_dump_json(exclude={"root"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) to the directory containing config/default.yaml."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / CONFIG_DIR_NAME / DEFAULT_CONFIG_NAME).is_file():
            return candidate
    raise ConfigError(
        f"Could not find {CONFIG_DIR_NAME}/{DEFAULT_CONFIG_NAME} in {here} or any parent. "
        "Run `radar` from inside the project, or pass --config."
    )


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Turn ``RADAR__A__B=value`` env vars into ``{"a": {"b": value}}``."""
    result: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.upper().startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue
        node = result
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"Conflicting environment overrides at {key}")
            node = child
        node[parts[-1]] = _parse_scalar(raw)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return data


def load_settings(
    config_path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    load_env_file: bool = True,
) -> Settings:
    """Load and validate settings.

    Args:
        config_path: explicit YAML file. Default: ``<project root>/config/default.yaml``.
        environ: environment mapping for overrides. Default: ``os.environ``.
        load_env_file: load ``<root>/.env`` into ``os.environ`` first (no override).
    """
    if config_path is None:
        root = find_project_root()
        config_path = root / CONFIG_DIR_NAME / DEFAULT_CONFIG_NAME
    else:
        config_path = config_path.resolve()
        if not config_path.is_file():
            raise ConfigError(f"Config file not found: {config_path}")
        parent = config_path.parent
        root = parent.parent if parent.name == CONFIG_DIR_NAME else parent

    env_file = root / ".env"
    if load_env_file and env_file.is_file():
        load_dotenv(env_file, override=False)

    env: Mapping[str, str] = os.environ if environ is None else environ

    data = _read_yaml(config_path)
    local_path = config_path.parent / LOCAL_CONFIG_NAME
    if local_path.is_file() and local_path != config_path:
        data = deep_merge(data, _read_yaml(local_path))
    data = deep_merge(data, env_overrides(env))

    try:
        return Settings.model_validate({**data, "root": root})
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc
