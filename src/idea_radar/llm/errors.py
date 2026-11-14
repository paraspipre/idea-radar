"""LLM error hierarchy. Callers branch on these types, never on message text."""

from __future__ import annotations

from datetime import datetime


class LLMError(Exception):
    """Base class for every LLM-layer failure."""


class LLMConfigError(LLMError):
    """Setup problem: not logged in, CLI missing, wrong backend, API key in env. Not retryable."""


class TransientLLMError(LLMError):
    """Server error, overload, timeout, or connection hiccup. Retried with backoff."""


class LLMOutputError(LLMError):
    """The model's output was missing or did not match the schema."""


class LLMRefusalError(LLMError):
    """The model declined the request."""


class StopRun(LLMError):
    """The run must stop now, save its state, and resume next time (exit code 75)."""


class BudgetExhausted(StopRun):
    """The per-run or per-stage call budget is used up, or a rate-limit warning asked us to stop."""


class RateLimited(StopRun):
    """The Claude subscription usage limit was hit."""

    def __init__(self, message: str, resets_at: datetime | None = None) -> None:
        super().__init__(message)
        self.resets_at = resets_at
