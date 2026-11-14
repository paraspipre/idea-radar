"""The backend interface: one structured-output call.

Caching, budgets, retries and logging live in LLMClient, not in backends.
"""

from __future__ import annotations

from typing import Any, Protocol

from idea_radar.llm.types import BackendResponse, EffortLevel, LLMRequest


class Backend(Protocol):
    name: str

    async def complete(
        self, request: LLMRequest[Any], *, model: str, effort: EffortLevel
    ) -> BackendResponse:
        """Run one call. Raise subclasses of ``LLMError`` on failure."""
        ...
