"""LLMClient: the single entry point for every model call in Idea Radar.

Per call: cache lookup -> budget check -> backend call (retrying transient errors with
backoff) -> schema validation -> cache write -> call log. Rate-limit signals feed the
governor so the run stops cleanly; StopRun subclasses tell the caller to exit with 75.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from idea_radar.config import LLMEffort, LLMModels
from idea_radar.db.models import LLMCallStatus
from idea_radar.llm.backend import Backend
from idea_radar.llm.cache import LLMCache, cache_key
from idea_radar.llm.call_log import CallLog, CallRecord
from idea_radar.llm.errors import LLMError, LLMOutputError, RateLimited, TransientLLMError
from idea_radar.llm.governor import BudgetGovernor
from idea_radar.llm.types import LLMRequest, LLMResult, LLMUsage, RateLimitSnapshot
from idea_radar.log import get_logger

Sleep = Callable[[float], Awaitable[None]]

_log = get_logger(component="llm")


class LLMClient:
    def __init__(
        self,
        *,
        backend: Backend,
        models: LLMModels,
        effort: LLMEffort,
        governor: BudgetGovernor,
        cache: LLMCache | None = None,
        call_log: CallLog | None = None,
        run_id: str | None = None,
        max_attempts: int = 3,
        backoff_base_s: float = 2.0,
        sleep: Sleep = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.backend = backend
        self.governor = governor
        self._models = models
        self._effort = effort
        self._cache = cache
        self._call_log = call_log
        self._run_id = run_id
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._sleep = sleep
        self._jitter = jitter

    async def complete_json[T: BaseModel](self, request: LLMRequest[T]) -> LLMResult[T]:
        model = self._models.strong if request.tier == "strong" else self._models.small
        effort = self._effort.strong if request.tier == "strong" else self._effort.small
        key = cache_key(backend=self.backend.name, model=model, effort=effort, request=request)
        log = _log.bind(
            stage=request.stage,
            prompt=f"{request.prompt_name}.v{request.prompt_version}",
            model=model,
            n_items=request.n_items,
        )

        if request.use_cache and self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                try:
                    data = request.schema.model_validate(cached)
                except ValidationError:
                    log.warning("llm_cache_entry_invalid", cache_key=key)
                else:
                    self._record(
                        request, model=model, status=LLMCallStatus.CACHE_HIT, attempt=0, key=key
                    )
                    log.info("llm_cache_hit")
                    return LLMResult(data=data, usage=LLMUsage(), from_cache=True, cache_key=key)

        attempt = 0
        while True:
            attempt += 1
            self.governor.ensure_can_call(request.stage)  # raises BudgetExhausted
            self.governor.record_call(request.stage)
            try:
                response = await self.backend.complete(request, model=model, effort=effort)
            except RateLimited as exc:
                self.governor.observe_rate_limit("rejected", exc.resets_at)
                self._record(
                    request,
                    model=model,
                    status=LLMCallStatus.RATE_LIMITED,
                    attempt=attempt,
                    key=key,
                    rate_limit=RateLimitSnapshot(status="rejected", resets_at=exc.resets_at),
                    error=str(exc),
                )
                log.warning("llm_rate_limited", resets_at=exc.resets_at, attempt=attempt)
                raise
            except TransientLLMError as exc:
                self._record(
                    request,
                    model=model,
                    status=LLMCallStatus.TRANSIENT_ERROR,
                    attempt=attempt,
                    key=key,
                    error=str(exc),
                )
                if attempt >= self._max_attempts:
                    log.error("llm_transient_giving_up", attempt=attempt, error=str(exc))
                    raise
                delay = self._backoff_base_s * (2 ** (attempt - 1)) * (1 + self._jitter())
                log.warning("llm_retry", attempt=attempt, delay_s=round(delay, 2), error=str(exc))
                await self._sleep(delay)
                continue
            except LLMError as exc:
                status = (
                    LLMCallStatus.OUTPUT_ERROR
                    if isinstance(exc, LLMOutputError)
                    else LLMCallStatus.ERROR
                )
                self._record(
                    request, model=model, status=status, attempt=attempt, key=key, error=str(exc)
                )
                log.error("llm_call_failed", attempt=attempt, error=str(exc))
                raise

            if response.rate_limit is not None:
                self.governor.observe_rate_limit(
                    response.rate_limit.status, response.rate_limit.resets_at
                )
            try:
                data = request.schema.model_validate(response.data)
            except ValidationError as exc:
                self._record(
                    request,
                    model=model,
                    status=LLMCallStatus.OUTPUT_ERROR,
                    attempt=attempt,
                    key=key,
                    usage=response.usage,
                    auth_source=response.auth_source,
                    rate_limit=response.rate_limit,
                    error=str(exc),
                )
                raise LLMOutputError(
                    f"Output for stage {request.stage!r} did not match "
                    f"{request.schema.__name__} ({exc.error_count()} errors)."
                ) from exc

            if request.use_cache and self._cache is not None:
                self._cache.put(
                    key,
                    stage=request.stage,
                    model=model,
                    prompt_name=request.prompt_name,
                    prompt_version=request.prompt_version,
                    response=data.model_dump(mode="json"),
                )
            self._record(
                request,
                model=model,
                status=LLMCallStatus.OK,
                attempt=attempt,
                key=key,
                usage=response.usage,
                auth_source=response.auth_source,
                rate_limit=response.rate_limit,
            )
            log.info(
                "llm_call_ok",
                attempt=attempt,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                duration_ms=response.usage.duration_ms,
                auth_source=response.auth_source,
            )
            return LLMResult(
                data=data,
                usage=response.usage,
                from_cache=False,
                cache_key=key,
                rate_limit=response.rate_limit,
                auth_source=response.auth_source,
            )

    def _record(
        self,
        request: LLMRequest[Any],
        *,
        model: str,
        status: LLMCallStatus,
        attempt: int,
        key: str,
        usage: LLMUsage | None = None,
        auth_source: str | None = None,
        rate_limit: RateLimitSnapshot | None = None,
        error: str | None = None,
    ) -> None:
        if self._call_log is None:
            return
        self._call_log.record(
            CallRecord(
                run_id=self._run_id,
                stage=request.stage,
                prompt=f"{request.prompt_name}.v{request.prompt_version}",
                backend=self.backend.name,
                model=model,
                status=status,
                attempt=attempt,
                n_items=request.n_items,
                usage=usage if usage is not None else LLMUsage(),
                cache_key=key,
                auth_source=auth_source,
                rate_limit=rate_limit,
                error=error,
            )
        )
