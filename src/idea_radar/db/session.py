"""Engine and session factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


def _sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def make_engine(url: str, *, echo: bool = False) -> Engine:
    """Create an engine; for SQLite, create the parent directory and set pragmas."""
    parsed = make_url(url)
    is_sqlite = parsed.get_backend_name() == "sqlite"
    if is_sqlite:
        database = parsed.database
        if database and database != ":memory:" and not database.startswith("file:"):
            Path(database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, echo=echo)
    if is_sqlite:
        event.listen(engine, "connect", _sqlite_pragmas)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def redact_url(url: str) -> str:
    """URL safe to print (password hidden; SQLite shown as a readable file path)."""
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite":
        return f"sqlite:///{parsed.database or ':memory:'}"
    return parsed.render_as_string(hide_password=True)
