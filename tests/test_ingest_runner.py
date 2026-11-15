"""run_ingest end to end with a fake source: cursors, dedupe, PII, errors, CLI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from idea_radar import cli
from idea_radar.config import Settings, load_settings
from idea_radar.db.models import IngestCursor, RawItem
from idea_radar.ingest import runner as runner_module
from idea_radar.ingest.http import PoliteClient, SourceError
from idea_radar.ingest.runner import parse_cursor, run_ingest
from idea_radar.ingest.types import FetchedItem, FetchResult, Source

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


def item(source_id: str, text: str, *, minutes: int = 0) -> FetchedItem:
    return FetchedItem(
        source="fake",
        source_id=source_id,
        url=f"https://example.test/{source_id}",
        title=None,
        text=text,
        created_at=NOW - timedelta(minutes=minutes),
        score=3,
    )


PROBLEM = (
    "I wish there was a tool to sync invoices automatically, I'd pay for it. "
    "Email me at a@b.co please."
)


class FakeSource:
    name = "fake"

    def __init__(self, batches: list[list[FetchedItem]], fail: bool = False) -> None:
        self.batches = batches
        self.fail = fail
        self.since_seen: list[datetime] = []

    def queries(self) -> list[str]:
        return ["q1"]

    def fetch(
        self, client: PoliteClient, query: str, *, since: datetime, limit: int
    ) -> FetchResult:
        self.since_seen.append(since)
        if self.fail:
            raise SourceError("boom")
        batch = self.batches.pop(0)
        cursor = max((i.created_at for i in batch), default=since).isoformat()
        return FetchResult(items=batch, cursor=cursor, requests=1)


@pytest.fixture
def settings(config_path: Path) -> Settings:
    return load_settings(config_path, environ={}, load_env_file=False)


def patch_sources(monkeypatch: pytest.MonkeyPatch, *sources: Source) -> None:
    def fake_build(
        settings: Settings, *, only: list[str] | None = None
    ) -> tuple[list[tuple[Source, float]], list[str]]:
        return [(s, 0.0) for s in sources], []

    monkeypatch.setattr(runner_module, "build_sources", fake_build)


def test_ingest_is_incremental_dedupes_and_scrubs(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:
    source = FakeSource(
        [
            [
                item("1", PROBLEM, minutes=5),
                item("2", "Nice weather today, nothing to report here at all, really.", minutes=4),
                item("3", PROBLEM.upper(), minutes=3),  # same content, different id
            ],
            [item("1", PROBLEM, minutes=5), item("4", PROBLEM + " Also CSV.", minutes=1)],
        ]
    )
    patch_sources(monkeypatch, source)

    first = run_ingest(settings, session_factory, now=lambda: NOW)
    (report,) = first.sources
    assert (report.new, report.passed, report.rejected, report.duplicates) == (3, 1, 1, 1)
    assert source.since_seen[0] == NOW - timedelta(days=settings.ingest.lookback_days)

    second = run_ingest(settings, session_factory, now=lambda: NOW)
    (report2,) = second.sources
    assert (report2.new, report2.updated, report2.passed) == (1, 1, 1)
    assert parse_cursor(source.since_seen[1].isoformat()) == NOW - timedelta(minutes=3)

    with session_factory() as session:
        rows = {r.source_id: r for r in session.scalars(select(RawItem))}
        cursor = session.get(IngestCursor, ("fake", "q1"))
    assert set(rows) == {"1", "2", "3", "4"}
    assert rows["3"].prefilter_status == "duplicate"
    assert "a@b.co" not in rows["1"].snippet and "[email]" in rows["1"].snippet
    assert "problem:wish" in rows["1"].prefilter_reasons
    assert cursor is not None


def test_failing_source_is_reported_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:
    good = FakeSource([[item("1", PROBLEM)]])
    bad = FakeSource([], fail=True)
    bad.name = "bad"
    patch_sources(monkeypatch, bad, good)
    report = run_ingest(settings, session_factory, now=lambda: NOW)
    assert report.sources[0].errors == ["q1: boom"]
    assert report.sources[1].new == 1
    assert not report.failed_completely


def test_cli_ingest_run_and_stats(monkeypatch: pytest.MonkeyPatch, config_path: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(cli.app, ["--config", str(config_path), "db", "init"]).exit_code == 0
    patch_sources(monkeypatch, FakeSource([[item("1", PROBLEM)]]))
    result = runner.invoke(cli.app, ["--config", str(config_path), "ingest", "run"])
    assert result.exit_code == 0, result.output
    assert "LLM calls: 0" in result.output
    stats = runner.invoke(cli.app, ["--config", str(config_path), "ingest", "stats"])
    assert stats.exit_code == 0, stats.output
    assert "fake" in stats.output and "I wish there was a tool" in stats.output

    bad = runner.invoke(cli.app, ["--config", str(config_path), "ingest", "run", "-s", "nope"])
    assert bad.exit_code == 2
