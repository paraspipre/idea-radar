"""Build the configured backend and LLMClient. The API-credential guard runs here."""

from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy.orm import Session, sessionmaker

from idea_radar.config import Settings
from idea_radar.llm.backend import Backend
from idea_radar.llm.cache import LLMCache
from idea_radar.llm.call_log import CallLog
from idea_radar.llm.client import LLMClient
from idea_radar.llm.governor import BudgetGovernor
from idea_radar.llm.guard import ensure_subscription_auth


def build_backend(settings: Settings, *, environ: Mapping[str, str] | None = None) -> Backend:
    ensure_subscription_auth(
        backend=settings.llm.backend,
        allow_api_key_env=settings.llm.allow_api_key_env,
        environ=os.environ if environ is None else environ,
    )
    if settings.llm.backend == "agent_sdk":
        # Imported lazily so nothing outside llm/agent_sdk.py loads the SDK.
        from idea_radar.llm.agent_sdk import AgentSDKBackend

        return AgentSDKBackend(
            cwd=settings.data_path / "llm_sandbox",
            timeout_s=settings.llm.timeout_s,
            max_budget_usd=settings.llm.max_budget_usd_per_call,
        )
    from idea_radar.llm.anthropic_api import AnthropicAPIBackend

    return AnthropicAPIBackend()


def build_governor(settings: Settings) -> BudgetGovernor:
    return BudgetGovernor(
        max_calls=settings.budget.max_llm_calls_per_run,
        per_stage=dict(settings.budget.per_stage),
        stop_on_warning=settings.budget.stop_on_rate_limit_warning,
    )


def build_client(
    settings: Settings,
    *,
    factory: sessionmaker[Session],
    run_id: str | None,
    backend: Backend | None = None,
    governor: BudgetGovernor | None = None,
) -> LLMClient:
    return LLMClient(
        backend=backend if backend is not None else build_backend(settings),
        models=settings.llm.models,
        effort=settings.llm.effort,
        governor=governor if governor is not None else build_governor(settings),
        cache=LLMCache(factory),
        call_log=CallLog(factory),
        run_id=run_id,
        max_attempts=settings.llm.max_attempts,
        backoff_base_s=settings.llm.backoff_base_s,
    )
