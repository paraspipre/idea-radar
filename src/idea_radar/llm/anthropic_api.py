"""Stub backend for the Anthropic API (pay-per-token with your own API key).

Not implemented yet: it exists so the LLMClient interface stays honest and forks have an
obvious place to plug in. Enable with `llm.backend: anthropic_api` once implemented.
"""

from __future__ import annotations

from typing import Any

from idea_radar.llm.errors import LLMConfigError
from idea_radar.llm.types import BackendResponse, EffortLevel, LLMRequest


class AnthropicAPIBackend:
    name = "anthropic_api"

    async def complete(
        self, request: LLMRequest[Any], *, model: str, effort: EffortLevel
    ) -> BackendResponse:
        raise LLMConfigError(
            "The anthropic_api backend is a stub. Use llm.backend: agent_sdk, or implement "
            "this class with the official `anthropic` SDK and your own API key."
        )
