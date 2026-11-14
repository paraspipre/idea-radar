from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.config import LLMEffort, LLMModels
from idea_radar.db.models import LLMCall
from idea_radar.llm.cache import LLMCache
from idea_radar.llm.call_log import CallLog
from idea_radar.llm.client import LLMClient
from idea_radar.llm.errors import (
    BudgetExhausted,
    LLMError,
    LLMOutputError,
    RateLimited,
    TransientLLMError,
)
from idea_radar.llm.fake import FakeBackend, FakeReply
from idea_radar.llm.governor import BudgetGovernor
from idea_radar.llm.types import LLMRequest, RateLimitSnapshot


class Answer(BaseModel):
    value: int


def make_request(
    prompt: str = "q", *, stage: str = "extract", version: int = 1
) -> LLMRequest[Answer]:
    return LLMRequest(
        stage=stage,
        prompt_name="test",
        prompt_version=version,
        system="sys",
        prompt=prompt,
        schema=Answer,
    )


class Harness:
    def __init__(
        self, factory: sessionmaker[Session], governor: BudgetGovernor | None = None
    ) -> None:
        self.backend = FakeBackend()
        self.sleeps: list[float] = []
        self.factory = factory

        async def fake_sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        self.client = LLMClient(
            backend=self.backend,
            models=LLMModels(small="haiku", strong="sonnet"),
            effort=LLMEffort(),
            governor=governor or BudgetGovernor(max_calls=10),
            cache=LLMCache(factory),
            call_log=CallLog(factory),
            run_id=None,
            max_attempts=3,
            backoff_base_s=1.0,
            sleep=fake_sleep,
            jitter=lambda: 0.0,
        )

    def call(self, request: LLMRequest[Answer]) -> Answer:
        return asyncio.run(self.client.complete_json(request)).data

    def statuses(self) -> list[str]:
        with self.factory() as session:
            return list(session.scalars(select(LLMCall.status).order_by(LLMCall.id)))


@pytest.fixture
def harness(session_factory: sessionmaker[Session]) -> Harness:
    return Harness(session_factory)


def test_success_is_validated_logged_and_cached(harness: Harness) -> None:
    harness.backend.queue({"value": 7})
    assert harness.call(make_request()).value == 7
    # identical request: served from cache, backend not called again, budget untouched
    assert harness.call(make_request()).value == 7
    assert len(harness.backend.calls) == 1
    assert harness.client.governor.calls == 1
    assert harness.statuses() == ["ok", "cache_hit"]


def test_prompt_version_change_misses_cache(harness: Harness) -> None:
    harness.backend.queue({"value": 1}, {"value": 2})
    assert harness.call(make_request(version=1)).value == 1
    assert harness.call(make_request(version=2)).value == 2
    assert len(harness.backend.calls) == 2


def test_tier_selects_model_and_effort(harness: Harness) -> None:
    harness.backend.queue({"value": 1})
    request = LLMRequest(
        stage="skeptic",
        prompt_name="t",
        prompt_version=1,
        system="s",
        prompt="p",
        schema=Answer,
        tier="strong",
    )
    asyncio.run(harness.client.complete_json(request))
    call = harness.backend.calls[0]
    assert (call.model, call.effort) == ("sonnet", "medium")


def test_schema_mismatch_raises_and_is_not_cached(harness: Harness) -> None:
    harness.backend.queue({"value": "not a number"}, {"value": 3})
    with pytest.raises(LLMOutputError):
        harness.call(make_request())
    assert harness.call(make_request()).value == 3  # not served from a bad cache entry
    assert harness.statuses() == ["output_error", "ok"]


def test_transient_errors_are_retried_with_backoff(harness: Harness) -> None:
    harness.backend.queue(
        FakeReply(error=TransientLLMError("overloaded")),
        FakeReply(error=TransientLLMError("overloaded")),
        {"value": 5},
    )
    assert harness.call(make_request()).value == 5
    assert harness.sleeps == [1.0, 2.0]
    assert harness.client.governor.calls == 3  # every attempt counts against the budget
    assert harness.statuses() == ["transient_error", "transient_error", "ok"]


def test_transient_errors_give_up_after_max_attempts(harness: Harness) -> None:
    harness.backend.queue(*[FakeReply(error=TransientLLMError("down")) for _ in range(3)])
    with pytest.raises(TransientLLMError):
        harness.call(make_request())
    assert len(harness.backend.calls) == 3


def test_non_transient_errors_are_not_retried(harness: Harness) -> None:
    harness.backend.queue(FakeReply(error=LLMError("bad request")))
    with pytest.raises(LLMError):
        harness.call(make_request())
    assert len(harness.backend.calls) == 1
    assert harness.sleeps == []


def test_stage_and_run_budgets_are_enforced(session_factory: sessionmaker[Session]) -> None:
    governor = BudgetGovernor(max_calls=3, per_stage={"extract": 2})
    h = Harness(session_factory, governor)
    h.backend.queue({"value": 1}, {"value": 2}, {"value": 3})
    h.call(make_request("a"))
    h.call(make_request("b"))
    with pytest.raises(BudgetExhausted, match="extract"):
        h.call(make_request("c"))
    h.call(make_request("d", stage="naming"))  # other stage still has run budget
    with pytest.raises(BudgetExhausted, match="Run budget"):
        h.call(make_request("e", stage="naming"))
    assert len(h.backend.calls) == 3


def test_rate_limit_warning_stops_the_next_call(harness: Harness) -> None:
    warning = RateLimitSnapshot(status="allowed_warning", utilization=0.85)
    harness.backend.queue(FakeReply(data={"value": 1}, rate_limit=warning), {"value": 2})
    assert harness.call(make_request("a")).value == 1  # the warned call itself succeeds
    with pytest.raises(BudgetExhausted, match="Approaching"):
        harness.call(make_request("b"))
    # cache hits still work after a stop: they cost nothing
    assert harness.call(make_request("a")).value == 1


def test_rate_limit_rejection_stops_and_records_reset(harness: Harness) -> None:
    resets = datetime(2026, 11, 16, 3, 0, tzinfo=UTC)
    harness.backend.queue(FakeReply(error=RateLimited("limit", resets_at=resets)))
    with pytest.raises(RateLimited):
        harness.call(make_request())
    assert harness.client.governor.resets_at == resets
    assert harness.statuses() == ["rate_limited"]
    with pytest.raises(BudgetExhausted):
        harness.call(make_request("other"))
