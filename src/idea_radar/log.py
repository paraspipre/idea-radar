"""Structured logging (structlog) on stderr, with run_id and stage bound from context.

stdout stays clean for command output.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal, cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["console", "json"]


def _stderr_logger_factory(*_args: Any) -> structlog.PrintLogger:
    # Look up sys.stderr for every new logger so redirected streams (pytest, CliRunner) work.
    return structlog.PrintLogger(file=sys.stderr)


def configure_logging(level: LogLevel = "INFO", fmt: LogFormat = "console") -> None:
    """Configure structlog (and stdlib logging for libraries). Calling again replaces it."""
    level_no = logging.getLevelNamesMapping()[level]
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if fmt == "json":
        processors.append(structlog.processors.dict_tracebacks)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=_stderr_logger_factory,
        cache_logger_on_first_use=False,
    )
    # Third-party libraries (alembic, httpx, ...) use stdlib logging.
    logging.basicConfig(
        level=level_no,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    # Alembic's per-command INFO chatter ("Context impl SQLiteImpl") is noise outside DEBUG.
    if level_no > logging.DEBUG:
        logging.getLogger("alembic").setLevel(logging.WARNING)


def get_logger(**initial_values: Any) -> FilteringBoundLogger:
    return cast(FilteringBoundLogger, structlog.get_logger(**initial_values))


def bind_context(*, run_id: str | None = None, stage: str | None = None) -> None:
    """Attach run_id and/or stage to every log line emitted from this context."""
    values = {key: value for key, value in (("run_id", run_id), ("stage", stage)) if value}
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
