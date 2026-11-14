"""Programmatic Alembic helpers, so `radar db init` works without an alembic.ini."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # ConfigParser interpolation: a literal '%' must be doubled.
    cfg.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))
    return cfg


def head_revision(db_url: str) -> str | None:
    return ScriptDirectory.from_config(alembic_config(db_url)).get_current_head()


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def upgrade_to_head(db_url: str) -> None:
    command.upgrade(alembic_config(db_url), "head")
