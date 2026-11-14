"""AgentSDKBackend tests with a stubbed `query()` that yields real SDK message objects."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    RateLimitEvent,
    RateLimitInfo,
    ResultError,
    ResultMessage,
    SystemMessage,
)
from pydantic import BaseModel

from idea_radar.llm import agent_sdk
from idea_radar.llm.agent_sdk import AgentSDKBackend, build_options
from idea_radar.llm.errors import (
    LLMConfigError,
    LLMOutputError,
    LLMRefusalError,
    RateLimited,
    TransientLLMError,
)
from idea_radar.llm.types import BackendResponse, LLMRequest


class Reply(BaseModel):
    ok: bool


REQUEST = LLMRequest(
    stage="smoke", prompt_name="smoke", prompt_version=1, system="sys", prompt="hi", schema=Reply
)


def result_message(**overrides: Any) -> ResultMessage:
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 1200,
        "duration_api_ms": 1000,
        "is_error": False,
        "num_turns": 2,
        "session_id": "s1",
        "structured_output": {"ok": True},
        "total_cost_usd": 0.0012,
        "model_usage": {
            "claude-haiku-5-5": {
                "inputTokens": 900,
                "outputTokens": 40,
                "cacheReadInputTokens": 3000,
                "cacheCreationInputTokens": 0,
                "webSearchRequests": 0,
                "costUSD": 0.0012,
                "contextWindow": 1_000_000,
                "maxOutputTokens": 64_000,
            }
        },
    }
    fields.update(overrides)
    return ResultMessage(**fields)


def init_message(source: str = "none") -> SystemMessage:
    return SystemMessage(subtype="init", data={"apiKeySource": source})


def rate_event(status: str, resets_at: int | None = 1_900_000_000) -> RateLimitEvent:
    info = RateLimitInfo(status=status, resets_at=resets_at, rate_limit_type="five_hour")  # type: ignore[arg-type]
    return RateLimitEvent(rate_limit_info=info, uuid="u", session_id="s1")


class StubQuery:
    def __init__(self, messages: list[Any], raise_after: Exception | None = None) -> None:
        self.messages = messages
        self.raise_after = raise_after
        self.options: ClaudeAgentOptions | None = None
        self.prompt: str | None = None

    async def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions, transport: Any = None
    ) -> AsyncIterator[Any]:
        self.prompt = prompt
        self.options = options
        for message in self.messages:
            yield message
        if self.raise_after is not None:
            raise self.raise_after


def run_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub: StubQuery
) -> BackendResponse:
    monkeypatch.setattr(agent_sdk, "query", stub)
    backend = AgentSDKBackend(cwd=tmp_path / "sandbox", timeout_s=30, max_budget_usd=0.5)
    return asyncio.run(backend.complete(REQUEST, model="haiku", effort="low"))


def test_options_are_token_lean_and_isolated(tmp_path: Path) -> None:
    options = build_options(REQUEST, model="haiku", effort="low", cwd=tmp_path, max_budget_usd=0.5)
    assert options.setting_sources == []  # no CLAUDE.md / settings / hooks
    assert options.tools == []  # no built-in tools
    assert options.allowed_tools == []
    assert options.system_prompt == "sys"
    assert options.cwd == tmp_path
    assert options.model == "haiku"
    assert options.effort == "low"
    assert options.max_turns == 3
    assert options.output_format is not None
    assert options.output_format["type"] == "json_schema"
    assert options.output_format["schema"]["properties"]["ok"]["type"] == "boolean"
    assert options.env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"


def test_success_maps_data_usage_auth_and_rate_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub = StubQuery([init_message("none"), rate_event("allowed_warning"), result_message()])
    response = run_backend(monkeypatch, tmp_path, stub)
    assert response.data == {"ok": True}
    assert response.auth_source == "none"
    assert response.usage.input_tokens == 900
    assert response.usage.cache_read_tokens == 3000
    assert response.usage.output_tokens == 40
    assert response.usage.models == ("claude-haiku-5-5",)
    assert response.usage.cost_usd_est == pytest.approx(0.0012)
    assert response.rate_limit is not None
    assert response.rate_limit.status == "allowed_warning"
    assert response.rate_limit.window == "five_hour"
    assert response.rate_limit.resets_at is not None
    assert stub.prompt == "hi"
    assert (tmp_path / "sandbox").is_dir()


def test_rejected_rate_limit_event_raises_rate_limited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub = StubQuery([rate_event("rejected"), result_message()])
    with pytest.raises(RateLimited) as excinfo:
        run_backend(monkeypatch, tmp_path, stub)
    assert excinfo.value.resets_at is not None


def test_result_error_429_maps_to_rate_limited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = {"subtype": "success", "is_error": True, "api_error_status": 429}
    stub = StubQuery(
        [result_message(is_error=True, api_error_status=429, structured_output=None)],
        raise_after=ResultError("failed", data=data),
    )
    with pytest.raises(RateLimited):
        run_backend(monkeypatch, tmp_path, stub)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"subtype": "success", "api_error_status": 529}, TransientLLMError),
        ({"subtype": "success", "terminal_reason": "api_error"}, TransientLLMError),
        ({"subtype": "success", "api_error_status": 401}, LLMConfigError),
        ({"subtype": "error_max_structured_output_retries"}, LLMOutputError),
        ({"subtype": "error_max_turns"}, LLMOutputError),
        ({"subtype": "success", "stop_reason": "refusal"}, LLMRefusalError),
    ],
)
def test_result_errors_are_classified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    data: dict[str, Any],
    expected: type[Exception],
) -> None:
    stub = StubQuery([], raise_after=ResultError("failed", data=data))
    with pytest.raises(expected):
        run_backend(monkeypatch, tmp_path, stub)


def test_missing_structured_output_is_an_output_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub = StubQuery([result_message(structured_output=None)])
    with pytest.raises(LLMOutputError, match="No structured output"):
        run_backend(monkeypatch, tmp_path, stub)
