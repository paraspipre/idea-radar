"""Structured logging (structlog). Logs go to stderr; stdout stays clean for command output."""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    level_no = logging.getLevelNamesMapping()[level.upper()]

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if fmt == "json":
        processors += [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
    # Third-party libraries (alembic, httpx, ...) use stdlib logging.
    logging.basicConfig(
        level=level_no,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def get_logger(**initial_values: Any) -> FilteringBoundLogger:
    return cast(FilteringBoundLogger, structlog.get_logger(**initial_values))
