"""Command-line interface: `radar ...`.

No `from __future__ import annotations` here: Typer inspects parameter annotations at runtime.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import typer
import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from idea_radar import __version__, exit_codes
from idea_radar.config import ConfigError, Settings, load_settings
from idea_radar.db.migrate import current_revision, head_revision, upgrade_to_head
from idea_radar.db.models import Run
from idea_radar.db.session import make_engine, redact_url
from idea_radar.doctor import Status, run_checks
from idea_radar.log import configure_logging

app = typer.Typer(
    name="radar",
    help="Idea Radar: find unmet problems and rank them as evidence-backed opportunities.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database schema management.", no_args_is_help=True)
app.add_typer(db_app, name="db")

_STATUS_COLORS = {
    Status.OK: typer.colors.GREEN,
    Status.WARN: typer.colors.YELLOW,
    Status.FAIL: typer.colors.RED,
}


@dataclass(frozen=True)
class CliState:
    config_path: Path | None


ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Config YAML to use (default: config/default.yaml in the project).",
    ),
]


@app.callback()
def main(ctx: typer.Context, config: ConfigOption = None) -> None:
    ctx.obj = CliState(config_path=config)


def _settings(ctx: typer.Context) -> Settings:
    state = cast(CliState, ctx.obj)
    try:
        settings = load_settings(state.config_path)
    except ConfigError as exc:
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=exit_codes.USAGE) from exc
    configure_logging(settings.app.log_level, settings.app.log_format)
    return settings


@app.command()
def version() -> None:
    """Print the Idea Radar version."""
    typer.echo(__version__)


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Check Python, config, LLM auth safety, data dir and database. Makes no LLM calls."""
    state = cast(CliState, ctx.obj)
    checks = run_checks(state.config_path, os.environ)
    width = max(len(check.name) for check in checks)
    for check in checks:
        typer.secho(f"[{check.status.value:^4}] ", fg=_STATUS_COLORS[check.status], nl=False)
        typer.echo(f"{check.name.ljust(width)}  {check.detail}")
    if any(check.status is Status.FAIL for check in checks):
        typer.secho("\nSome checks failed.", fg=typer.colors.RED)
        raise typer.Exit(code=exit_codes.FAILURE)
    typer.secho("\nNo failures.", fg=typer.colors.GREEN)


@app.command("config")
def show_config(ctx: typer.Context) -> None:
    """Print the effective config (defaults + local.yaml + RADAR__ env overrides)."""
    settings = _settings(ctx)
    typer.echo(yaml.safe_dump(settings.model_dump(mode="json"), sort_keys=False).rstrip())
    typer.echo(f"# project root: {settings.root}")
    typer.echo(f"# data dir:     {settings.data_path}")
    typer.echo(f"# database:     {redact_url(settings.db_url)}")
    typer.echo(f"# fingerprint:  {settings.fingerprint()[:12]}")


@db_app.command("init")
def db_init(ctx: typer.Context) -> None:
    """Create the database, or upgrade it to the latest migration. Safe to re-run."""
    settings = _settings(ctx)
    upgrade_to_head(settings.db_url)
    engine = make_engine(settings.db_url)
    try:
        revision = current_revision(engine)
    finally:
        engine.dispose()
    typer.secho(
        f"Database ready at revision {revision}: {redact_url(settings.db_url)}",
        fg=typer.colors.GREEN,
    )


@db_app.command("status")
def db_status(ctx: typer.Context) -> None:
    """Show the current and latest schema revisions and how many runs are recorded."""
    settings = _settings(ctx)
    engine = make_engine(settings.db_url)
    try:
        current = current_revision(engine)
        head = head_revision(settings.db_url)
        typer.echo(f"Database: {redact_url(settings.db_url)}")
        typer.echo(f"Current:  {current or '(not initialized)'}")
        typer.echo(f"Latest:   {head}")
        if current != head:
            typer.secho("Run: uv run radar db init", fg=typer.colors.YELLOW)
            raise typer.Exit(code=exit_codes.FAILURE)
        with Session(engine) as session:
            runs = session.scalar(select(func.count()).select_from(Run)) or 0
        typer.echo(f"Runs:     {runs}")
    finally:
        engine.dispose()
