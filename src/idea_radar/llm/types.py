"""Value types shared by the LLM client and its backends."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from idea_radar.config import EffortLevel

ModelTier = Literal["small", "strong"]
RateLimitStatus = Literal["allowed", "allowed_warning", "rejected"]

__all__ = [
    "BackendResponse",
    "EffortLevel",
    "LLMRequest",
    "LLMResult",
    "LLMUsage",
    "ModelTier",
    "RateLimitSnapshot",
    "RateLimitStatus",
]


@dataclass(frozen=True)
class LLMRequest[T: BaseModel]:
    """One structured-output LLM call.

    ``stage`` is the budget bucket and log label (e.g. "extract"). ``system`` should come from a
    versioned prompt file; ``prompt_name``/``prompt_version`` identify it in logs and cache keys.
    """

    stage: str
    prompt_name: str
    prompt_version: int
    system: str
    prompt: str
    schema: type[T]
    tier: ModelTier = "small"
    tools: tuple[str, ...] = ()
    max_turns: int = 3  # structured output arrives via a tool call, so >= 2
    n_items: int = 1  # items batched into this call (for the budget panel)
    use_cache: bool = True


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd_est: float | None = None  # SDK's API-equivalent estimate (not billed on Pro)
    duration_ms: int = 0
    num_turns: int = 0
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class RateLimitSnapshot:
    status: RateLimitStatus
    resets_at: datetime | None = None
    utilization: float | None = None
    window: str | None = None  # e.g. "five_hour", "seven_day"


@dataclass(frozen=True)
class BackendResponse:
    data: dict[str, Any]
    usage: LLMUsage
    rate_limit: RateLimitSnapshot | None = None
    auth_source: str | None = None  # which credential paid (SDK init `apiKeySource`)


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    data: T
    usage: LLMUsage
    from_cache: bool
    cache_key: str
    rate_limit: RateLimitSnapshot | None = None
    auth_source: str | None = None
