"""Command-line interface: `radar ...`.

No `from __future__ import annotations` here: Typer inspects parameter annotations at runtime.
"""

import asyncio
import json
import math
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, cast

import typer
import yaml
from pydantic import BaseModel
from sqlalchemy import Engine, case, func, select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar import __version__, exit_codes
from idea_radar.clock import utc_now
from idea_radar.config import ConfigError, Settings, load_settings
from idea_radar.db.migrate import current_revision, head_revision, upgrade_to_head
from idea_radar.db.models import LLMCall, LLMCallStatus, Run, RunKind, RunStatus
from idea_radar.db.runs import finish_run, rate_limit_block, start_run
from idea_radar.db.session import make_engine, make_session_factory, redact_url
from idea_radar.doctor import Status, run_checks
from idea_radar.embed import Embedder, FastEmbedder
from idea_radar.evals.clusters import (
    PAIRS_FILE,
    append_pair,
    candidate_pairs,
    cluster_quality,
    latest_ranked_clusters,
    load_pairs,
)
from idea_radar.evals.gold import (
    append_gold,
    candidates_to_label,
    gold_from_row,
    gold_path,
    load_gold,
)
from idea_radar.evals.metrics import extraction_metrics, fmt, prefilter_metrics
from idea_radar.ingest.runner import run_ingest
from idea_radar.ingest.sources import ALL_SOURCES
from idea_radar.ingest.store import IngestStore
from idea_radar.llm.client import LLMClient
from idea_radar.llm.errors import LLMConfigError, LLMError, RateLimited, StopRun
from idea_radar.llm.factory import build_backend, build_client
from idea_radar.llm.governor import BudgetGovernor
from idea_radar.llm.prompts import load_prompt
from idea_radar.llm.types import LLMRequest, LLMResult
from idea_radar.log import configure_logging
from idea_radar.pipeline.cluster import cluster_members_text, reset_clusters, run_clustering
from idea_radar.pipeline.extract import (
    ExtractedProblem,
    LLMItem,
    extract_items,
    extractor_version,
    pending_items,
    render_items,
    run_extraction,
)
from idea_radar.pipeline.graph import EXTRA_STAGES, PipelineContext, pipeline_stages, run_pipeline

app = typer.Typer(
    name="radar",
    help="Idea Radar: find unmet problems and rank them as evidence-backed opportunities.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database schema management.", no_args_is_help=True)
app.add_typer(db_app, name="db")
llm_app = typer.Typer(help="LLM layer utilities.", no_args_is_help=True)
app.add_typer(llm_app, name="llm")
ingest_app = typer.Typer(
    help="Stage 1: fetch, clean and prefilter items (no LLM).", no_args_is_help=True
)
app.add_typer(ingest_app, name="ingest")
eval_app = typer.Typer(help="Evals: hand labels and metrics.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")

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


def _open_db(settings: Settings) -> tuple[Engine, sessionmaker[Session]]:
    """Engine + session factory, or exit with a hint if the schema is not at the latest revision."""
    engine = make_engine(settings.db_url)
    if current_revision(engine) != head_revision(settings.db_url):
        engine.dispose()
        typer.secho("Database is not up to date. Run: uv run radar db init", fg=typer.colors.YELLOW)
        raise typer.Exit(code=exit_codes.FAILURE)
    return engine, make_session_factory(engine)


class Tier(StrEnum):
    small = "small"
    strong = "strong"


class SmokeReply(BaseModel):
    ok: bool
    echo: str


SMOKE_PROMPT = "Set ok to true and echo to the single word: radar"


@llm_app.command("smoke")
def llm_smoke(
    ctx: typer.Context,
    tier: Annotated[Tier, typer.Option(help="Model tier to test.")] = Tier.small,
) -> None:
    """Make ONE tiny real LLM call: checks subscription auth and measures per-call overhead."""
    settings = _settings(ctx)
    engine, factory = _open_db(settings)
    try:
        blocked_until = rate_limit_block(factory, now=utc_now())
        if blocked_until is not None:
            typer.secho(
                f"Usage limit active until {blocked_until:%Y-%m-%d %H:%M} UTC; not calling.",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=exit_codes.TEMPFAIL)
        try:
            backend = build_backend(settings)
        except LLMConfigError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=exit_codes.FAILURE) from exc

        run_id = start_run(factory, kind=RunKind.SMOKE.value, config_hash=settings.fingerprint())
        client = build_client(settings, factory=factory, run_id=run_id, backend=backend)
        prompt = load_prompt("smoke", 1)
        request = LLMRequest(
            stage="smoke",
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            system=prompt.text,
            prompt=SMOKE_PROMPT,
            schema=SmokeReply,
            tier="strong" if tier is Tier.strong else "small",
            use_cache=False,
        )
        try:
            result = asyncio.run(client.complete_json(request))
        except StopRun as exc:
            limited = isinstance(exc, RateLimited)
            finish_run(
                factory,
                run_id,
                status=RunStatus.RATE_LIMITED if limited else RunStatus.BUDGET_EXHAUSTED,
                calls_used=client.governor.calls,
                resets_at=client.governor.resets_at,
                notes=str(exc),
            )
            typer.secho(f"Stopped: {exc}", fg=typer.colors.YELLOW)
            raise typer.Exit(code=exit_codes.TEMPFAIL) from exc
        except LLMError as exc:
            finish_run(
                factory,
                run_id,
                status=RunStatus.ERROR,
                calls_used=client.governor.calls,
                notes=str(exc),
            )
            typer.secho(f"LLM call failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=exit_codes.FAILURE) from exc

        finish_run(
            factory,
            run_id,
            status=RunStatus.COMPLETED,
            calls_used=client.governor.calls,
            resets_at=client.governor.resets_at,
        )
        _print_smoke(result, prompt_chars=len(prompt.text) + len(SMOKE_PROMPT))
    finally:
        engine.dispose()


def _print_smoke(result: LLMResult[SmokeReply], *, prompt_chars: int) -> None:
    usage = result.usage
    ok = result.data.ok and result.data.echo.strip().lower() == "radar"
    typer.secho(
        f"Reply:          ok={result.data.ok} echo={result.data.echo!r}",
        fg=typer.colors.GREEN if ok else typer.colors.YELLOW,
    )
    typer.echo(f"Models:         {', '.join(usage.models) or '(not reported)'}")
    auth = result.auth_source or "(not reported)"
    typer.echo(f"Auth source:    {auth}")
    if "API_KEY" in auth.upper() or "AUTH_TOKEN" in auth.upper():
        typer.secho(
            "WARNING: this call was paid by an API credential, not your subscription!",
            fg=typer.colors.RED,
        )
    typer.echo(
        f"Input tokens:   {usage.input_tokens}  (our prompt is ~{prompt_chars // 4}; "
        "the rest is fixed per-call overhead)"
    )
    typer.echo(
        f"Cache tokens:   read={usage.cache_read_tokens} created={usage.cache_creation_tokens}"
    )
    typer.echo(f"Output tokens:  {usage.output_tokens}")
    typer.echo(f"Turns:          {usage.num_turns}   Duration: {usage.duration_ms / 1000:.1f}s")
    if usage.cost_usd_est is not None:
        typer.echo(f"API-equivalent: ${usage.cost_usd_est:.4f} (estimate; not billed on Pro)")
    if result.rate_limit is not None:
        rl = result.rate_limit
        resets = f", resets {rl.resets_at:%Y-%m-%d %H:%M} UTC" if rl.resets_at else ""
        util = f", {rl.utilization:.0%} used" if rl.utilization is not None else ""
        typer.echo(f"Rate limit:     {rl.status} ({rl.window or 'window?'}{util}{resets})")


@app.command()
def budget(
    ctx: typer.Context,
    runs: Annotated[int, typer.Option(min=1, help="How many recent runs to show.")] = 10,
) -> None:
    """Recent runs with LLM calls, cache hits, tokens and API-equivalent cost. No LLM calls."""
    settings = _settings(ctx)
    engine, factory = _open_db(settings)
    try:
        real_call = case((LLMCall.status != LLMCallStatus.CACHE_HIT.value, 1), else_=0)
        cache_hit = case((LLMCall.status == LLMCallStatus.CACHE_HIT.value, 1), else_=0)
        stmt = (
            select(
                Run,
                func.coalesce(func.sum(real_call), 0),
                func.coalesce(func.sum(cache_hit), 0),
                func.coalesce(func.sum(LLMCall.input_tokens), 0),
                func.coalesce(func.sum(LLMCall.output_tokens), 0),
                func.coalesce(func.sum(LLMCall.cost_usd_est), 0.0),
            )
            .outerjoin(LLMCall, LLMCall.run_id == Run.id)
            .group_by(Run.id)
            .order_by(Run.started_at.desc())
            .limit(runs)
        )
        with factory() as session:
            rows = session.execute(stmt).all()
        if not rows:
            typer.echo("No runs yet.")
        else:
            header = (
                f"{'started (UTC)':<17} {'kind':<9} {'status':<17} "
                f"{'calls':>5} {'hits':>5} {'in tok':>8} {'out tok':>8} {'est $':>8}"
            )
            typer.echo(header)
            typer.echo("-" * len(header))
            for run, calls, hits, tok_in, tok_out, cost in rows:
                typer.echo(
                    f"{run.started_at:%Y-%m-%d %H:%M} {run.kind:<9} {run.status:<17} "
                    f"{calls:>5} {hits:>5} {tok_in:>8} {tok_out:>8} {cost:>8.4f}"
                )
        blocked_until = rate_limit_block(factory, now=utc_now())
        if blocked_until is not None:
            typer.secho(
                f"\nUsage limit active until {blocked_until:%Y-%m-%d %H:%M} UTC.",
                fg=typer.colors.YELLOW,
            )
    finally:
        engine.dispose()


SourceOption = Annotated[
    list[str] | None,
    typer.Option("--source", "-s", help=f"Only these sources: {', '.join(ALL_SOURCES)}."),
]


@ingest_app.command("run")
def ingest_run(ctx: typer.Context, source: SourceOption = None) -> None:
    """Fetch new items from every enabled source since its cursor, prefilter and store them."""
    settings = _settings(ctx)
    unknown = sorted(set(source or []) - set(ALL_SOURCES))
    if unknown:
        typer.secho(f"Unknown source(s): {', '.join(unknown)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=exit_codes.USAGE)
    engine, factory = _open_db(settings)
    try:
        report = run_ingest(settings, factory, only=source)
    finally:
        engine.dispose()

    header = (
        f"{'source':<14} {'queries':>7} {'reqs':>5} {'fetched':>7} {'new':>5} "
        f"{'passed':>6} {'rejected':>8} {'dupes':>5} {'updated':>7}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for s in report.sources:
        typer.echo(
            f"{s.source:<14} {s.queries:>7} {s.requests:>5} {s.fetched:>7} {s.new:>5} "
            f"{s.passed:>6} {s.rejected:>8} {s.duplicates:>5} {s.updated:>7}"
        )
        for error in s.errors:
            typer.secho(f"  ! {error[:160]}", fg=typer.colors.YELLOW)
    for note in report.notes:
        typer.secho(f"note: {note}", fg=typer.colors.BRIGHT_BLACK)
    if report.purged:
        typer.echo(f"Purged {report.purged} old rejected/duplicate items (retention policy).")
    typer.echo("LLM calls: 0")
    if report.failed_completely:
        typer.secho("Every source failed.", fg=typer.colors.RED)
        raise typer.Exit(code=exit_codes.FAILURE)


@ingest_app.command("stats")
def ingest_stats(
    ctx: typer.Context,
    top: Annotated[int, typer.Option(min=0, help="Show the N highest-scoring passed items.")] = 10,
) -> None:
    """Counts per source and prefilter status, plus the top passed items. No LLM calls."""
    settings = _settings(ctx)
    engine, factory = _open_db(settings)
    try:
        store = IngestStore(factory)
        counts = store.counts()
        best = store.top_passed(top) if top else []
    finally:
        engine.dispose()
    if not counts:
        typer.echo("No items yet. Run: uv run radar ingest run")
        return
    table: dict[str, dict[str, int]] = {}
    for src, status, n in counts:
        table.setdefault(src, {})[status] = n
    typer.echo(f"{'source':<14} {'passed':>7} {'rejected':>9} {'duplicate':>10} {'total':>7}")
    for src, by_status in sorted(table.items()):
        passed = by_status.get("passed", 0)
        rejected = by_status.get("rejected", 0)
        dupes = by_status.get("duplicate", 0)
        total = passed + rejected + dupes
        typer.echo(f"{src:<14} {passed:>7} {rejected:>9} {dupes:>10} {total:>7}")
    if best:
        typer.echo("")
        typer.echo(f"Top {len(best)} passed items (what the LLM will see first; $ = pays):")
        for item in best:
            label = (item.title or item.snippet)[:80]
            flag = " $" if item.has_wtp else ""
            typer.echo(f"  {item.prefilter_score:>5.2f}{flag:<2} [{item.source}] {label}")
            typer.secho(f"         {item.url}", fg=typer.colors.BRIGHT_BLACK)


def _begin_llm_run(
    settings: Settings,
    factory: sessionmaker[Session],
    kind: RunKind,
    *,
    governor: BudgetGovernor | None = None,
) -> tuple[str, LLMClient]:
    """Honour a pending usage-limit reset, build the guarded backend, open a run."""
    blocked_until = rate_limit_block(factory, now=utc_now())
    if blocked_until is not None:
        typer.secho(
            f"Usage limit active until {blocked_until:%Y-%m-%d %H:%M} UTC; not calling.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=exit_codes.TEMPFAIL)
    try:
        backend = build_backend(settings)
    except LLMConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=exit_codes.FAILURE) from exc
    run_id = start_run(factory, kind=kind.value, config_hash=settings.fingerprint())
    client = build_client(
        settings, factory=factory, run_id=run_id, backend=backend, governor=governor
    )
    return run_id, client


def _end_llm_run(
    factory: sessionmaker[Session], run_id: str, client: LLMClient, stop: StopRun | None
) -> None:
    """Close the run; exit 75 if it stopped on budget or a usage limit."""
    if stop is None:
        finish_run(
            factory,
            run_id,
            status=RunStatus.COMPLETED,
            calls_used=client.governor.calls,
            resets_at=client.governor.resets_at,
        )
        return
    limited = isinstance(stop, RateLimited)
    finish_run(
        factory,
        run_id,
        status=RunStatus.RATE_LIMITED if limited else RunStatus.BUDGET_EXHAUSTED,
        calls_used=client.governor.calls,
        resets_at=client.governor.resets_at,
        notes=str(stop),
    )
    typer.secho(f"Stopped early: {stop} Resume with the same command.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=exit_codes.TEMPFAIL)


def _fail_llm_run(
    factory: sessionmaker[Session], run_id: str, client: LLMClient, exc: LLMError
) -> None:
    finish_run(
        factory, run_id, status=RunStatus.ERROR, calls_used=client.governor.calls, notes=str(exc)
    )
    typer.secho(f"LLM error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=exit_codes.FAILURE) from exc


@app.command()
def extract(
    ctx: typer.Context,
    max_batches: Annotated[
        int | None, typer.Option(min=1, help="Cap batches (LLM calls) for this run.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be sent; make no LLM calls.")
    ] = False,
) -> None:
    """Stage 2: extract problems from prefilter-passed items, ~25 per LLM call."""
    settings = _settings(ctx)
    engine, factory = _open_db(settings)
    try:
        size = settings.extraction.batch_size
        total_cap = settings.budget.max_llm_calls_per_run
        stage_cap = min(settings.budget.per_stage.get("extract", total_cap), total_cap)
        cap = min(max_batches, stage_cap) if max_batches else stage_cap
        items = pending_items(factory, version=extractor_version(settings))
        needed = math.ceil(len(items) / size)
        typer.echo(
            f"Pending items: {len(items)} -> {needed} batch(es) of {size}; "
            f"this run will do at most {min(needed, cap)}."
        )
        if dry_run:
            prompt = load_prompt("extract", settings.extraction.prompt_version)
            first = render_items(items[:size])
            tokens = (len(prompt.text) + len(first)) // 4
            per_item = tokens // max(1, len(items[:size]))
            typer.echo(
                f"Estimated input tokens for the first batch: ~{tokens} "
                f"(~{per_item} per item, plus fixed per-call overhead)"
            )
            typer.echo("")
            typer.echo("--- system prompt ---")
            typer.echo(prompt.text)
            typer.echo("--- first batch (user message, truncated) ---")
            typer.echo(first[:3000])
            typer.echo("LLM calls: 0 (dry run)")
            return
        if not items:
            typer.echo("Nothing to extract. Run `radar ingest run` first.")
            return
        run_id, client = _begin_llm_run(settings, factory, RunKind.PIPELINE)
        try:
            report = asyncio.run(
                run_extraction(settings, factory, client, run_id=run_id, max_batches=cap)
            )
        except LLMError as exc:
            _fail_llm_run(factory, run_id, client, exc)
            return
        typer.echo(
            f"Processed {report.processed} items: {report.problems} problems "
            f"({report.quotes_verified} with verified quotes), {report.not_problems} not "
            f"problems, {report.failed} failed. LLM calls: {client.governor.calls}."
        )
        _end_llm_run(factory, run_id, client, report.stop)
    finally:
        engine.dispose()


@eval_app.command("label")
def eval_label(
    ctx: typer.Context,
    n: Annotated[int, typer.Option(min=1, help="How many items to label this session.")] = 20,
    seed: Annotated[int | None, typer.Option(help="Shuffle seed (reproducible order).")] = None,
) -> None:
    """Hand-label items (problem? willing to pay?) into evals/data/extraction_gold.jsonl."""
    settings = _settings(ctx)
    path = gold_path(settings)
    existing = {g.id for g in load_gold(path)}
    engine, factory = _open_db(settings)
    try:
        rows = candidates_to_label(factory, exclude=existing, n=n, seed=seed)
    finally:
        engine.dispose()
    if not rows:
        typer.echo("Nothing new to label. Run `radar ingest run` to fetch more items.")
        return
    typer.echo(f"Labeling {len(rows)} items -> {path}  (already labeled: {len(existing)})")
    typer.echo("A 'problem' = someone describes a real need or pain a product could solve.")
    saved = 0
    for index, row in enumerate(rows, start=1):
        typer.echo("")
        typer.secho(f"[{index}/{len(rows)}] {row.source} | {row.context or ''}", bold=True)
        if row.title:
            typer.echo(f"Title: {row.title}")
        typer.echo(row.snippet)
        typer.secho(row.url, fg=typer.colors.BRIGHT_BLACK)
        answer = str(typer.prompt("Problem? [y]es / [n]o / [s]kip / [q]uit", default="s"))
        answer = answer.strip().lower()
        if answer.startswith("q"):
            break
        if not answer.startswith(("y", "n")):
            continue
        is_problem = answer.startswith("y")
        wtp = False
        if is_problem:
            wtp_answer = str(typer.prompt("Willing to pay / price complaint? [y/n]", default="n"))
            wtp = wtp_answer.strip().lower().startswith("y")
        append_gold(path, gold_from_row(row, is_problem=is_problem, wtp=wtp, note=None))
        saved += 1
    typer.echo(f"Saved {saved} labels. Total gold items: {len(existing) + saved}.")


@eval_app.command("extraction")
def eval_extraction(
    ctx: typer.Context,
    yes: Annotated[
        bool, typer.Option("--yes", help="Allow the LLM calls (cached after the first run).")
    ] = False,
    prefilter_only: Annotated[
        bool, typer.Option("--prefilter-only", help="Only the free prefilter metrics.")
    ] = False,
) -> None:
    """Prefilter recall (free) and extraction precision/recall (LLM) against the gold labels."""
    settings = _settings(ctx)
    gold = load_gold(gold_path(settings))
    if not gold:
        typer.secho("No gold labels yet. Run: uv run radar eval label", fg=typer.colors.YELLOW)
        raise typer.Exit(code=exit_codes.FAILURE)
    positives = sum(1 for g in gold if g.is_problem)
    typer.echo(f"Gold items: {len(gold)} ({positives} problems, {len(gold) - positives} not)")
    pf = prefilter_metrics(gold)
    results: dict[str, object] = {
        "date": utc_now().isoformat(),
        "n_gold": len(gold),
        "n_problems": positives,
        "prefilter": pf.as_dict(),
    }
    typer.echo(
        f"Prefilter (free): recall {fmt(pf.recall)}  precision {fmt(pf.precision)}  "
        f"(missed {pf.fn} real problems; let through {pf.fp} non-problems)"
    )

    calls_needed = math.ceil(len(gold) / settings.extraction.batch_size)
    if not prefilter_only and not yes:
        typer.secho(
            f"Extraction eval needs ~{calls_needed} LLM call(s) (free if cached). "
            "Re-run with --yes to allow them.",
            fg=typer.colors.YELLOW,
        )
    elif not prefilter_only:
        engine, factory = _open_db(settings)
        try:
            governor = BudgetGovernor(
                max_calls=calls_needed * 2,
                stop_on_warning=settings.budget.stop_on_rate_limit_warning,
            )
            run_id, client = _begin_llm_run(settings, factory, RunKind.EVAL, governor=governor)
            items = [
                LLMItem(
                    id=g.id, source=g.source, snippet=g.snippet, title=g.title, context=g.context
                )
                for g in gold
            ]
            prompt = load_prompt("extract", settings.extraction.prompt_version)
            try:
                outcome = asyncio.run(
                    extract_items(client, items, prompt=prompt, settings=settings)
                )
            except StopRun as exc:
                _end_llm_run(factory, run_id, client, exc)
                return
            except LLMError as exc:
                _fail_llm_run(factory, run_id, client, exc)
                return
            _end_llm_run(factory, run_id, client, None)
        finally:
            engine.dispose()
        predictions: dict[str, ExtractedProblem | None] = dict(outcome.problems)
        predictions.update({item_id: None for item_id in outcome.not_problems})
        problem_m, wtp_m = extraction_metrics(gold, predictions)
        results["extraction"] = {
            "prompt": f"extract.v{settings.extraction.prompt_version}",
            "model": settings.llm.models.small,
            "batch_size": settings.extraction.batch_size,
            "llm_calls": client.governor.calls,
            "failed_items": len(outcome.failed),
            "is_problem": problem_m.as_dict(),
            "wtp_strong": wtp_m.as_dict(),
        }
        typer.echo(
            f"Extraction: precision {fmt(problem_m.precision)}  recall {fmt(problem_m.recall)}  "
            f"F1 {fmt(problem_m.f1)}   WTP precision {fmt(wtp_m.precision)}  "
            f"recall {fmt(wtp_m.recall)}   (LLM calls: {client.governor.calls})"
        )
    out = settings.root / "evals" / "results" / "extraction_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    typer.echo(f"Saved {out}")


def _make_embedder(settings: Settings) -> Embedder:
    return FastEmbedder(
        settings.clustering.embedding_model, cache_dir=str(settings.data_path / "models")
    )


StageChoice = Annotated[
    str, typer.Option("--until", help="Stop after this stage (default: run every stage).")
]


@app.command("run")
def run_command(
    ctx: typer.Context,
    until: StageChoice = "",
    skip: Annotated[
        list[str] | None, typer.Option("--skip", help="Skip a stage (repeatable).")
    ] = None,
) -> None:
    """Run the pipeline graph (stages 2-8) under the call budget. Resumable: just run it again."""
    settings = _settings(ctx)
    until = until or pipeline_stages()[-1]
    unknown = sorted({until, *(skip or [])} - set(pipeline_stages()))
    if unknown:
        typer.secho(f"Unknown stage(s): {', '.join(unknown)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=exit_codes.USAGE)
    engine, factory = _open_db(settings)
    try:
        run_id, client = _begin_llm_run(settings, factory, RunKind.PIPELINE)
        context = PipelineContext(
            settings=settings,
            factory=factory,
            client=client,
            embedder=_make_embedder(settings),
            run_id=run_id,
            until=until,
            skip=frozenset(skip or []),
            extra_stages=list(EXTRA_STAGES),
        )
        try:
            state = asyncio.run(run_pipeline(context))
        except LLMError as exc:
            _fail_llm_run(factory, run_id, client, exc)
            return
        for stage, summary in state.get("summary", {}).items():
            details = ", ".join(f"{k}={v}" for k, v in summary.items() if k != "top")
            typer.echo(f"{stage:<12} {details}")
        typer.echo(f"LLM calls this run: {client.governor.calls}")
        if context.ranked:
            typer.echo("")
            typer.echo("Top clusters by demand:")
            for r in context.ranked[: settings.demand.top_n + 2]:
                typer.echo(f"  {r.score:.3f}  ({r.size:>2} items)  {r.name or '(unnamed)'}")
        _end_llm_run(factory, run_id, client, context.stop)
    finally:
        engine.dispose()


@app.command("clusters")
def clusters_command(
    ctx: typer.Context,
    top: Annotated[int, typer.Option(min=1, help="How many clusters to show.")] = 10,
    members: Annotated[bool, typer.Option("--members", help="Show member problems.")] = False,
) -> None:
    """Clusters ranked by their latest demand score, with the score breakdown. No LLM calls."""
    settings = _settings(ctx)
    engine, factory = _open_db(settings)
    try:
        ranked = latest_ranked_clusters(factory, limit=top)
        if not ranked:
            typer.echo("No scored clusters yet. Run: uv run radar run")
            return
        for row in ranked:
            norm = row["components"]["normalized"]
            parts = "  ".join(f"{k[:5]}={v:.2f}" for k, v in norm.items())
            typer.secho(
                f"{row['score']:.3f}  [{row['size']} items, {row['n_sources']} sources]  "
                f"{row['name'] or '(unnamed)'}",
                bold=True,
            )
            typer.echo(f"        {parts}")
            if members:
                for m in cluster_members_text(factory, row["cluster_id"])[:6]:
                    typer.echo(f"        - [{m['source']}] {m['who']}: {m['problem']}")
    finally:
        engine.dispose()


@eval_app.command("label-pairs")
def eval_label_pairs(
    ctx: typer.Context,
    n: Annotated[int, typer.Option(min=1, help="How many pairs to label.")] = 30,
    seed: Annotated[int | None, typer.Option(help="Shuffle seed.")] = None,
) -> None:
    """Hand-label problem pairs (same underlying problem?) into evals/data/cluster_pairs.jsonl."""
    settings = _settings(ctx)
    path = settings.root / PAIRS_FILE
    done = {(p.a, p.b) for p in load_pairs(path)}
    engine, factory = _open_db(settings)
    try:
        pairs = candidate_pairs(
            factory, model=settings.clustering.embedding_model, exclude=done, n=n, seed=seed
        )
    finally:
        engine.dispose()
    if not pairs:
        typer.echo("No candidate pairs yet. Run `radar run` so there are clusters first.")
        return
    saved = 0
    for index, (a, b) in enumerate(pairs, start=1):
        typer.echo("")
        typer.secho(f"[{index}/{len(pairs)}]", bold=True)
        typer.echo(f"A: {a.text}")
        typer.echo(f"B: {b.text}")
        answer = str(typer.prompt("Same underlying problem? [y]es/[n]o/[s]kip/[q]uit", default="s"))
        answer = answer.strip().lower()
        if answer.startswith("q"):
            break
        if not answer.startswith(("y", "n")):
            continue
        append_pair(path, a, b, same=answer.startswith("y"))
        saved += 1
    typer.echo(f"Saved {saved} pair labels to {path}.")


@eval_app.command("clusters")
def eval_clusters(ctx: typer.Context) -> None:
    """Cluster quality: pairwise P/R/F1 vs labeled pairs, plus noise ratio and silhouette. Free."""
    settings = _settings(ctx)
    engine, factory = _open_db(settings)
    try:
        result = cluster_quality(
            factory,
            model=settings.clustering.embedding_model,
            pairs=load_pairs(settings.root / PAIRS_FILE),
        )
    finally:
        engine.dispose()
    typer.echo(
        f"Problems: {result['n_problems']}  clustered: {result['n_clustered']}  "
        f"clusters: {result['n_clusters']}  noise ratio: {fmt(result['noise_ratio'])}  "
        f"silhouette: {fmt(result['silhouette'])}"
    )
    pairwise = result.get("pairwise")
    if pairwise:
        typer.echo(
            f"Pairwise vs {pairwise['tp'] + pairwise['fp'] + pairwise['fn'] + pairwise['tn']} "
            f"labeled pairs: precision {fmt(pairwise['precision'])}  "
            f"recall {fmt(pairwise['recall'])}  F1 {fmt(pairwise['f1'])}"
        )
    else:
        typer.secho("No labeled pairs yet: uv run radar eval label-pairs", fg=typer.colors.YELLOW)
    out = settings.root / "evals" / "results" / "clusters_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    typer.echo(f"Saved {out}")


@app.command("recluster")
def recluster_command(
    ctx: typer.Context,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm deleting the current clusters.")
    ] = False,
) -> None:
    """Rebuild clusters from stored embeddings (after tuning thresholds). Problems are kept."""
    settings = _settings(ctx)
    if not yes:
        typer.secho(
            "This deletes all clusters, memberships and demand scores (they are recomputed by "
            "`radar run --skip extract`). Re-run with --yes.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=exit_codes.USAGE)
    engine, factory = _open_db(settings)
    try:
        removed = reset_clusters(factory)
        report = run_clustering(settings, factory, _make_embedder(settings))
    finally:
        engine.dispose()
    typer.echo(
        f"Removed {removed} clusters. " + ", ".join(f"{k}={v}" for k, v in vars(report).items())
    )
    typer.echo("Next: uv run radar run --skip extract   (names clusters, scores demand)")
