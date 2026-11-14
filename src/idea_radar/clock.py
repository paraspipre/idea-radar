"""Time helpers. All timestamps in Idea Radar are timezone-aware UTC."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)
