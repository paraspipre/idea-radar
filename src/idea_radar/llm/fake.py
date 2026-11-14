"""Scripted backend for tests: never touches the network or a model."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from idea_radar.llm.types import (
    BackendResponse,
    EffortLevel,
    LLMRequest,
    LLMUsage,
    RateLimitSnapshot,
)


@dataclass(frozen=True)
class FakeReply:
    data: dict[str, Any] | None = None
    error: Exception | None = None
    rate_limit: RateLimitSnapshot | None = None
    usage: LLMUsage = field(default_factory=lambda: LLMUsage(input_tokens=100, output_tokens=20))


@dataclass(frozen=True)
class FakeCall:
    request: LLMRequest[Any]
    model: str
    effort: EffortLevel


class FakeBackend:
    name = "fake"

    def __init__(self, replies: Iterable[FakeReply | dict[str, Any]] = ()) -> None:
        self._replies: deque[FakeReply] = deque(
            r if isinstance(r, FakeReply) else FakeReply(data=r) for r in replies
        )
        self.calls: list[FakeCall] = []

    def queue(self, *replies: FakeReply | dict[str, Any]) -> None:
        for reply in replies:
            self._replies.append(reply if isinstance(reply, FakeReply) else FakeReply(data=reply))

    async def complete(
        self, request: LLMRequest[Any], *, model: str, effort: EffortLevel
    ) -> BackendResponse:
        self.calls.append(FakeCall(request=request, model=model, effort=effort))
        if not self._replies:
            raise AssertionError(f"FakeBackend has no reply queued for stage {request.stage!r}")
        reply = self._replies.popleft()
        if reply.error is not None:
            raise reply.error
        return BackendResponse(
            data=dict(reply.data or {}),
            usage=reply.usage,
            rate_limit=reply.rate_limit,
            auth_source="fake",
        )
