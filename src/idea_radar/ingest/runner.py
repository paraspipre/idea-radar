"""Stage 1 orchestration: fetch every enabled source/query from its cursor, clean, prefilter, store.

Idempotent and incremental: re-running only fetches what is newer than each saved cursor,
and items already stored are refreshed (engagement), never duplicated. No LLM calls.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from idea_radar.clock import utc_now
from idea_radar.config import Settings
from idea_radar.db.models import RunKind, RunStatus
from idea_radar.db.runs import finish_run, start_run
from idea_radar.ingest.http import PoliteClient, SourceError
from idea_radar.ingest.prefilter import evaluate
from idea_radar.ingest.sources import build_sources
from idea_radar.ingest.store import IngestStore, PreparedItem
from idea_radar.ingest.text import (
    content_hash,
    html_to_text,
    is_probably_english,
    scrub_pii,
    snippet_around,
)
from idea_radar.ingest.types import FetchedItem
from idea_radar.log import get_logger

_log = get_logger(component="ingest")

ClientFactory = Callable[[float], PoliteClient]


@dataclass
class SourceReport:
    source: str
    queries: int = 0
    requests: int = 0
    fetched: int = 0
    new: int = 0
    updated: int = 0
    passed: int = 0
    rejected: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class IngestReport:
    run_id: str
    sources: list[SourceReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    purged: int = 0

    @property
    def failed_completely(self) -> bool:
        return bool(self.sources) and all(s.errors and s.fetched == 0 for s in self.sources)


def parse_cursor(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.isdigit():
        return datetime.fromtimestamp(int(value), UTC)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def prepare(item: FetchedItem, *, query: str, settings: Settings) -> PreparedItem:
    """Clean HTML, scrub PII, prefilter, and cut the snippet the LLM will later see."""
    pf = settings.prefilter
    title = scrub_pii(html_to_text(item.title)) if item.title else None
    body = scrub_pii(html_to_text(item.text))
    if title and body.startswith(title):
        body = body[len(title) :].lstrip(" .:-")
    combined = f"{title}. {body}" if title and body else (title or body)
    result = evaluate(
        combined,
        source=item.source,
        score=item.score,
        num_comments=item.num_comments,
        threshold=pf.threshold,
        min_chars=pf.min_chars,
        source_prior=pf.source_prior,
        focus_keywords=settings.focus.keywords,
    )
    offset = len(title) + 2 if title and body else 0
    position = None if result.first_signal_at is None else max(0, result.first_signal_at - offset)
    snippet = snippet_around(body or title or "", position, pf.snippet_chars)
    return PreparedItem(
        id=item.item_id,
        source=item.source,
        source_id=item.source_id,
        query=query,
        url=item.url,
        title=title,
        snippet=snippet,
        context=scrub_pii(item.context) if item.context else None,
        created_at=item.created_at,
        score=item.score,
        num_comments=item.num_comments,
        lang="en" if is_probably_english(combined) else "other",
        content_hash=content_hash(combined),
        prefilter_score=result.score,
        passed=result.passed,
        reasons=result.reasons,
        has_wtp=result.has_wtp,
    )


def run_ingest(
    settings: Settings,
    factory: sessionmaker[Session],
    *,
    only: list[str] | None = None,
    client_factory: ClientFactory | None = None,
    now: Callable[[], datetime] = utc_now,
) -> IngestReport:
    def default_client(interval: float) -> PoliteClient:
        return PoliteClient(
            user_agent=settings.ingest.user_agent,
            timeout_s=settings.ingest.request_timeout_s,
            min_interval_s=interval,
        )

    make_client = client_factory or default_client
    store = IngestStore(factory)
    run_id = start_run(factory, kind=RunKind.INGEST.value, config_hash=settings.fingerprint())
    sources, notes = build_sources(settings, only=only)
    report = IngestReport(run_id=run_id, notes=notes)
    started = now()
    default_since = started - timedelta(days=settings.ingest.lookback_days)

    try:
        for source, interval in sources:
            src_report = SourceReport(source=source.name)
            report.sources.append(src_report)
            with make_client(interval) as client:
                for query in source.queries():
                    src_report.queries += 1
                    since = parse_cursor(store.get_cursor(source.name, query)) or default_since
                    try:
                        fetched = source.fetch(
                            client, query, since=since, limit=settings.ingest.max_items_per_query
                        )
                    except (SourceError, KeyError, ValueError, TypeError) as exc:
                        message = f"{query}: {exc}"
                        src_report.errors.append(message)
                        _log.warning("ingest_query_failed", source=source.name, error=message)
                        continue
                    src_report.requests += fetched.requests
                    src_report.fetched += len(fetched.items)
                    prepared = [prepare(i, query=query, settings=settings) for i in fetched.items]
                    saved = store.save_items(prepared)
                    src_report.new += saved.new
                    src_report.updated += saved.updated
                    src_report.passed += saved.passed
                    src_report.rejected += saved.rejected
                    src_report.duplicates += saved.duplicates
                    if fetched.cursor:
                        store.set_cursor(source.name, query, fetched.cursor)
                    _log.info(
                        "ingest_query_done",
                        source=source.name,
                        query=query,
                        fetched=len(fetched.items),
                        new=saved.new,
                        passed=saved.passed,
                    )
        report.purged = store.purge(
            older_than=started - timedelta(days=settings.ingest.retention_days)
        )
    except BaseException as exc:
        finish_run(factory, run_id, status=RunStatus.ERROR, calls_used=0, notes=repr(exc)[:500])
        raise
    finish_run(
        factory,
        run_id,
        status=RunStatus.ERROR if report.failed_completely else RunStatus.COMPLETED,
        calls_used=0,
        notes=json.dumps([asdict(s) for s in report.sources])[:4000],
    )
    return report
