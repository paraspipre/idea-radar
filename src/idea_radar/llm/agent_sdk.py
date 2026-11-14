"""Default backend: the Claude Agent SDK on the user's Claude subscription login.

This is the ONLY module that imports ``claude_agent_sdk`` (verified against v0.2.182).

Token hygiene for every call:
- ``setting_sources=[]``: no CLAUDE.md, settings, hooks or skills are loaded or billed.
- ``cwd`` is an empty sandbox directory, never the repo.
- ``tools=[]`` unless the request asks for tools (only the competition stage does).
- a short custom ``system_prompt`` replaces the SDK default.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLIConnectionError,
    CLINotFoundError,
    RateLimitEvent,
    RateLimitInfo,
    ResultError,
    ResultMessage,
    SystemMessage,
    query,
)

from idea_radar import __version__
from idea_radar.llm.errors import (
    LLMConfigError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    RateLimited,
    TransientLLMError,
)
from idea_radar.llm.types import (
    BackendResponse,
    EffortLevel,
    LLMRequest,
    LLMUsage,
    RateLimitSnapshot,
)

TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 500, 502, 503, 504, 529})
AUTH_HTTP_STATUSES = frozenset({401, 403})

SUBPROCESS_ENV = {
    "CLAUDE_AGENT_SDK_CLIENT_APP": f"idea-radar/{__version__}",
    # Belt and braces next to setting_sources=[]: no CLAUDE.md, no skill lists or reminders.
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1",
}


def build_options(
    request: LLMRequest[Any],
    *,
    model: str,
    effort: EffortLevel,
    cwd: Path,
    max_budget_usd: float | None,
) -> ClaudeAgentOptions:
    tools = list(request.tools)
    return ClaudeAgentOptions(
        model=model,
        effort=effort,
        system_prompt=request.system,
        tools=tools,
        allowed_tools=tools,
        setting_sources=[],
        cwd=cwd,
        max_turns=request.max_turns,
        max_budget_usd=max_budget_usd,
        output_format={"type": "json_schema", "schema": request.schema.model_json_schema()},
        env=dict(SUBPROCESS_ENV),
    )


def rate_limit_snapshot(info: RateLimitInfo | None) -> RateLimitSnapshot | None:
    if info is None:
        return None
    resets_at = datetime.fromtimestamp(info.resets_at, UTC) if info.resets_at else None
    return RateLimitSnapshot(
        status=info.status,
        resets_at=resets_at,
        utilization=info.utilization,
        window=info.rate_limit_type,
    )


def usage_from_result(result: ResultMessage) -> LLMUsage:
    """Token usage across all models the CLI used for this call (prefers ``model_usage``)."""
    model_usage = result.model_usage or {}
    if model_usage:
        values = list(model_usage.values())
        input_tokens = sum(int(v.get("inputTokens", 0)) for v in values)
        output_tokens = sum(int(v.get("outputTokens", 0)) for v in values)
        cache_read = sum(int(v.get("cacheReadInputTokens", 0)) for v in values)
        cache_creation = sum(int(v.get("cacheCreationInputTokens", 0)) for v in values)
        cost: float | None = sum(float(v.get("costUSD", 0.0)) for v in values)
        models = tuple(sorted(model_usage))
    else:
        usage = result.usage or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cache_read = int(usage.get("cache_read_input_tokens", 0))
        cache_creation = int(usage.get("cache_creation_input_tokens", 0))
        cost = None
        models = ()
    if result.total_cost_usd is not None:
        cost = result.total_cost_usd
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        cost_usd_est=cost,
        duration_ms=result.duration_ms,
        num_turns=result.num_turns,
        models=models,
    )


@dataclass
class _Collected:
    """Everything we care about from one query()'s message stream."""

    result: ResultMessage | None = None
    rate_limit: RateLimitInfo | None = None
    auth_source: str | None = None
    assistant_error: str | None = None

    def observe(self, message: object) -> None:
        if isinstance(message, RateLimitEvent):
            self.rate_limit = message.rate_limit_info
        elif isinstance(message, SystemMessage) and message.subtype == "init":
            source = message.data.get("apiKeySource")
            self.auth_source = source if isinstance(source, str) else None
        elif isinstance(message, AssistantMessage) and message.error:
            self.assistant_error = message.error
        elif isinstance(message, ResultMessage):
            self.result = message


def classify_failure(
    *,
    collected: _Collected,
    subtype: str | None,
    api_error_status: int | None,
    terminal_reason: str | None,
    stop_reason: str | None,
    detail: str,
) -> LLMError:
    """Map a failed run to our error hierarchy (most specific first)."""
    snapshot = rate_limit_snapshot(collected.rate_limit)
    if (
        api_error_status == 429
        or collected.assistant_error == "rate_limit"
        or (snapshot is not None and snapshot.status == "rejected")
    ):
        resets_at = snapshot.resets_at if snapshot else None
        until = f" until {resets_at.isoformat()}" if resets_at else ""
        return RateLimited(f"Claude usage limit reached{until}.", resets_at=resets_at)
    if (
        collected.assistant_error in ("authentication_failed", "billing_error")
        or api_error_status in AUTH_HTTP_STATUSES
    ):
        return LLMConfigError(
            "Not authenticated with your Claude subscription. Run `claude` once in a terminal, "
            f"log in with /login, then retry. ({detail})"
        )
    if stop_reason == "refusal":
        return LLMRefusalError(f"The model declined the request. ({detail})")
    if subtype == "error_max_structured_output_retries":
        return LLMOutputError(f"No schema-valid output after the SDK's retries. ({detail})")
    if subtype == "error_max_turns":
        return LLMOutputError(f"Hit max_turns before producing output. ({detail})")
    if subtype == "error_max_budget_usd":
        return LLMError(f"Per-call cost ceiling (max_budget_usd_per_call) exceeded. ({detail})")
    if (
        api_error_status in TRANSIENT_HTTP_STATUSES
        or collected.assistant_error == "server_error"
        or (
            terminal_reason == "api_error" and (api_error_status is None or api_error_status >= 500)
        )
    ):
        return TransientLLMError(f"Transient API error (status={api_error_status}). ({detail})")
    return LLMError(
        f"Agent SDK run failed (subtype={subtype}, status={api_error_status}, "
        f"reason={terminal_reason}). ({detail})"
    )


def response_from(collected: _Collected) -> BackendResponse:
    snapshot = rate_limit_snapshot(collected.rate_limit)
    result = collected.result
    if snapshot is not None and snapshot.status == "rejected":
        raise classify_failure(
            collected=collected,
            subtype=None,
            api_error_status=None,
            terminal_reason=None,
            stop_reason=None,
            detail="rate-limit event",
        )
    if result is None:
        raise LLMError("The Agent SDK finished without a result message.")
    if result.is_error or result.stop_reason == "refusal":
        raise classify_failure(
            collected=collected,
            subtype=result.subtype,
            api_error_status=result.api_error_status,
            terminal_reason=result.terminal_reason,
            stop_reason=result.stop_reason,
            detail="; ".join(result.errors or []) or (result.result or "")[:300],
        )
    data = result.structured_output
    if not isinstance(data, dict):
        raise LLMOutputError(
            f"No structured output returned (subtype={result.subtype}, "
            f"stop_reason={result.stop_reason}, turns={result.num_turns})."
        )
    return BackendResponse(
        data=data,
        usage=usage_from_result(result),
        rate_limit=snapshot,
        auth_source=collected.auth_source,
    )


class AgentSDKBackend:
    name = "agent_sdk"

    def __init__(self, *, cwd: Path, timeout_s: float, max_budget_usd: float | None) -> None:
        self._cwd = cwd
        self._timeout_s = timeout_s
        self._max_budget_usd = max_budget_usd

    async def complete(
        self, request: LLMRequest[Any], *, model: str, effort: EffortLevel
    ) -> BackendResponse:
        self._cwd.mkdir(parents=True, exist_ok=True)
        options = build_options(
            request,
            model=model,
            effort=effort,
            cwd=self._cwd,
            max_budget_usd=self._max_budget_usd,
        )
        collected = _Collected()
        try:
            async with asyncio.timeout(self._timeout_s):
                async for message in query(prompt=request.prompt, options=options):
                    collected.observe(message)
        except TimeoutError as exc:
            raise TransientLLMError(f"Call timed out after {self._timeout_s:.0f}s.") from exc
        except ResultError as exc:
            stop_reason = exc.data.get("stop_reason")
            raise classify_failure(
                collected=collected,
                subtype=exc.subtype,
                api_error_status=exc.api_error_status,
                terminal_reason=exc.terminal_reason,
                stop_reason=stop_reason if isinstance(stop_reason, str) else None,
                detail="; ".join(exc.errors) or (exc.result or "")[:300],
            ) from exc
        except CLINotFoundError as exc:
            raise LLMConfigError(
                f"Claude Code CLI not found (it ships with claude-agent-sdk): {exc}"
            ) from exc
        except CLIConnectionError as exc:
            raise TransientLLMError(f"Could not talk to the Claude Code CLI: {exc}") from exc
        except ClaudeSDKError as exc:
            if collected.assistant_error is not None:
                raise classify_failure(
                    collected=collected,
                    subtype=None,
                    api_error_status=None,
                    terminal_reason=None,
                    stop_reason=None,
                    detail=str(exc)[:300],
                ) from exc
            raise LLMError(f"Agent SDK failure: {exc}") from exc
        return response_from(collected)
