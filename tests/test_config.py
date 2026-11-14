from __future__ import annotations

from pathlib import Path

import pytest

from idea_radar.config import ConfigError, Settings, deep_merge, env_overrides, load_settings


def _load(config_path: Path, environ: dict[str, str] | None = None) -> Settings:
    return load_settings(config_path, environ=environ or {}, load_env_file=False)


def test_default_config_loads(config_path: Path, project: Path) -> None:
    settings = _load(config_path)
    assert settings.llm.backend == "agent_sdk"
    assert settings.llm.allow_api_key_env is False
    assert settings.budget.max_llm_calls_per_run == 21
    assert sum(settings.budget.per_stage.values()) <= settings.budget.max_llm_calls_per_run
    assert settings.root.resolve() == project.resolve()
    assert settings.data_path == (project / "data").resolve()
    assert settings.db_url.startswith("sqlite:///")
    assert settings.db_url.endswith("/radar.db")


def test_local_yaml_overrides_defaults(config_path: Path) -> None:
    (config_path.parent / "local.yaml").write_text(
        "budget:\n  max_llm_calls_per_run: 30\n", encoding="utf-8"
    )
    settings = _load(config_path)
    assert settings.budget.max_llm_calls_per_run == 30
    # untouched siblings survive the merge
    assert settings.budget.per_stage["extract"] == 10
    assert settings.llm.models.small == "haiku"


def test_env_overrides_win_and_are_typed(config_path: Path) -> None:
    settings = _load(
        config_path,
        {
            "RADAR__LLM__MODELS__SMALL": "sonnet",
            "RADAR__BUDGET__MAX_LLM_CALLS_PER_RUN": "40",
            "RADAR__APP__LOG_FORMAT": "json",
        },
    )
    assert settings.llm.models.small == "sonnet"
    assert settings.budget.max_llm_calls_per_run == 40
    assert settings.app.log_format == "json"


def test_unknown_key_is_rejected(config_path: Path) -> None:
    with pytest.raises(ConfigError):
        _load(config_path, {"RADAR__LLM__TYPO": "1"})


def test_stage_budget_cannot_exceed_total(config_path: Path) -> None:
    with pytest.raises(ConfigError, match="exceeds"):
        _load(config_path, {"RADAR__BUDGET__MAX_LLM_CALLS_PER_RUN": "5"})


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        _load(tmp_path / "nope.yaml")


def test_fingerprint_is_stable_and_sensitive(config_path: Path) -> None:
    a = _load(config_path).fingerprint()
    b = _load(config_path).fingerprint()
    c = _load(config_path, {"RADAR__LLM__MODELS__STRONG": "opus"}).fingerprint()
    assert a == b
    assert a != c


def test_absolute_data_dir_is_respected(config_path: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    settings = _load(config_path, {"RADAR__APP__DATA_DIR": str(target)})
    assert settings.data_path.resolve() == target.resolve()


def test_deep_merge_and_env_parsing() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert env_overrides({"RADAR__X__Y": "true", "OTHER": "1"}) == {"x": {"y": True}}
