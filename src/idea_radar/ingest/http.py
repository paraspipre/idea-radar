"""A polite HTTP client: one minimum interval per source, retries with Retry-After."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from idea_radar.log import get_logger

_log = get_logger(component="ingest.http")

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class SourceError(Exception):
    """A source could not be fetched (after retries)."""


class PoliteClient:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout_s: float,
        min_interval_s: float = 1.0,
        max_attempts: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_s,
            follow_redirects=True,
            transport=transport,
        )
        self.min_interval_s = min_interval_s
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request: float | None = None
        self.requests = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def pause(self, seconds: float) -> None:
        """Extra wait requested by an API (e.g. Stack Exchange `backoff`)."""
        if seconds > 0:
            self._sleep(min(seconds, 120.0))

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval_s - (self._monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._monotonic()

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        last_error = ""
        for attempt in range(1, self._max_attempts + 1):
            self._throttle()
            self.requests += 1
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                _log.warning("http_error", url=url, attempt=attempt, error=last_error)
                self._sleep(2.0**attempt)
                continue
            if response.status_code in RETRY_STATUSES and attempt < self._max_attempts:
                delay = _retry_after(response) or 2.0**attempt
                _log.warning("http_retry", url=url, status=response.status_code, delay_s=delay)
                self._sleep(min(delay, 120.0))
                continue
            if response.status_code >= 400:
                raise SourceError(
                    f"GET {url} -> HTTP {response.status_code}: {response.text[:200]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise SourceError(f"GET {url} returned invalid JSON") from exc
        raise SourceError(f"GET {url} failed after {self._max_attempts} attempts: {last_error}")


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
