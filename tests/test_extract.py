"""Stage 2 extraction with FakeBackend: rendering, mapping, quotes, idempotency, stops, CLI."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from idea_radar import cli, exit_codes
from idea_radar.config import Settings, load_settings
from idea_radar.db.models import Extraction, PrefilterStatus, Problem, RawItem
from idea_radar.evals.gold import GoldItem
from idea_radar.evals.metrics import Binary, extraction_metrics, prefilter_metrics
from idea_radar.llm.errors import LLMOutputError, RateLimited
from idea_radar.llm.factory import build_client
from idea_radar.llm.fake import FakeBackend, FakeReply
from idea_radar.pipeline.extract import (
    ExtractedProblem,
    LLMItem,
    extractor_version,
    quote_in,
    render_items,
    run_extraction,
)

NOW = datetime(2026, 11, 15, tzinfo=UTC)


def problem(
    i: int, quote: str = "sync invoices by hand every week", wtp: str = "strong"
) -> dict[str, Any]:
    return {
        "i": i,
        "who": "freelancers",
        "problem": "Reconciling invoices takes hours.",
        "workaround": "spreadsheets",
        "urgency": 4,
        "wtp": wtp,
        "quote": quote,
    }


def add_items(factory: sessionmaker[Session], n: int, *, status: str = "passed") -> list[str]:
    ids = []
    with factory() as session:
        for k in range(n):
            item_id = f"item{k:03d}"
            ids.append(item_id)
            session.add(
                RawItem(
                    id=item_id,
                    source="hackernews",
                    source_id=str(k),
                    query="q",
                    url=f"https://example.test/{k}",
                    title=None,
                    snippet=f"Item {k}: I have to sync invoices by hand every week, I'd pay.",
                    context="Ask HN",
                    created_at=NOW,
                    content_hash=f"h{k}",
                    prefilter_score=float(100 - k),
                    prefilter_status=status,
                    prefilter_reasons=[],
                )
            )
        session.commit()
    return ids


@pytest.fixture
def settings(config_path: Path) -> Settings:
    settings = load_settings(config_path, environ={}, load_env_file=False)
    return settings.model_copy(
        update={"extraction": settings.extraction.model_copy(update={"batch_size": 3})}
    )


def test_render_items_is_compact_and_indexed() -> None:
    text = render_items(
        [
            LLMItem(
                id="a", source="appstore", snippet="Keeps crashing", title="Bad", context="Slack"
            ),
            LLMItem(id="b", source="hackernews", snippet="Is there a tool?"),
        ]
    )
    assert text.splitlines() == [
        "Items:",
        "[0] (appstore | Slack) Bad. Keeps crashing",
        "[1] (hackernews) Is there a tool?",
    ]


def test_quote_verification() -> None:
    item = LLMItem(id="a", source="s", snippet="I have to sync invoices by hand -- every week!")
    assert quote_in("sync invoices by hand every week", item)
    assert quote_in("Sync invoices by hand, every week", item)
    assert not quote_in("export invoices to CSV automatically", item)
    assert not quote_in("week", item)  # too short to count as evidence


def test_run_extraction_saves_batches_and_never_reprocesses(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    add_items(session_factory, 5)
    backend = FakeBackend(
        [
            {"problems": [problem(0), problem(2, quote="made up words never said")]},
            {"problems": [problem(1, wtp="none"), problem(9)]},  # index 9 is out of range
        ]
    )
    client = build_client(settings, factory=session_factory, run_id=None, backend=backend)
    report = asyncio.run(run_extraction(settings, session_factory, client, run_id=None))
    assert (report.processed, report.problems, report.not_problems) == (5, 3, 2)
    assert report.quotes_verified == 2
    assert len(backend.calls) == 2
    assert "[2] (hackernews | Ask HN)" in backend.calls[0].request.prompt

    with session_factory() as session:
        problems = {p.raw_item_id: p for p in session.scalars(select(Problem))}
        statuses = {e.raw_item_id: e.status for e in session.scalars(select(Extraction))}
    assert set(problems) == {"item000", "item002", "item004"}
    assert problems["item002"].quote is None and not problems["item002"].quote_verified
    assert problems["item000"].wtp_strength == 1.0
    assert statuses["item001"] == "not_problem"

    again = asyncio.run(run_extraction(settings, session_factory, client, run_id=None))
    assert again.pending == 0 and len(backend.calls) == 2  # nothing is ever sent twice


def test_output_error_splits_the_batch_once(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    add_items(session_factory, 3)
    backend = FakeBackend(
        [
            FakeReply(error=LLMOutputError("bad schema")),
            {"problems": [problem(0)]},  # first half (1 item)
            FakeReply(error=LLMOutputError("still bad")),  # second half (2 items) -> failed
        ]
    )
    client = build_client(settings, factory=session_factory, run_id=None, backend=backend)
    report = asyncio.run(run_extraction(settings, session_factory, client, run_id=None))
    assert (report.problems, report.failed) == (1, 2)
    assert [len(c.request.prompt.splitlines()) - 1 for c in backend.calls] == [3, 1, 2]


def test_stop_mid_run_keeps_finished_batches(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    add_items(session_factory, 6)
    backend = FakeBackend(
        [{"problems": [problem(0)]}, FakeReply(error=RateLimited("limit", resets_at=NOW))]
    )
    client = build_client(settings, factory=session_factory, run_id=None, backend=backend)
    report = asyncio.run(run_extraction(settings, session_factory, client, run_id=None))
    assert report.processed == 3 and isinstance(report.stop, RateLimited)
    with session_factory() as session:
        assert len(list(session.scalars(select(Extraction)))) == 3


def test_extractor_version_tracks_prompt_and_model(settings: Settings) -> None:
    assert extractor_version(settings) == "extract.v1:haiku"


def gold(item_id: str, *, is_problem: bool, wtp: bool = False, passed: bool = True) -> GoldItem:
    return GoldItem(
        id=item_id,
        source="hackernews",
        url="u",
        snippet="s",
        prefilter_status="passed" if passed else "rejected",
        prefilter_score=2.0,
        is_problem=is_problem,
        wtp=wtp,
        labeled_at=NOW,
    )


def test_metrics() -> None:
    b = Binary.from_pairs([(True, True), (True, False), (False, True), (False, False)])
    assert (b.tp, b.fp, b.fn, b.tn) == (1, 1, 1, 1)
    assert b.precision == 0.5 and b.recall == 0.5 and b.f1 == 0.5
    assert Binary().precision is None

    items = [
        gold("a", is_problem=True, wtp=True),
        gold("b", is_problem=True, passed=False),
        gold("c", is_problem=False),
    ]
    pf = prefilter_metrics(items)
    assert pf.recall == 0.5 and pf.precision == 0.5
    predicted = ExtractedProblem.model_validate(problem(0))
    problem_m, wtp_m = extraction_metrics(items, {"a": predicted, "c": None})
    assert (problem_m.tp, problem_m.tn) == (1, 1)
    assert wtp_m.tp == 1


def test_cli_extract_dry_run_label_and_eval(
    monkeypatch: pytest.MonkeyPatch, config_path: Path, project: Path
) -> None:
    runner = CliRunner()
    base = ["--config", str(config_path)]
    assert runner.invoke(cli.app, [*base, "db", "init"]).exit_code == 0
    settings = load_settings(config_path, environ={}, load_env_file=False)
    from idea_radar.db.session import make_engine, make_session_factory

    engine = make_engine(settings.db_url)
    try:
        add_items(make_session_factory(engine), 2)
        add_items_rejected = make_session_factory(engine)
        with add_items_rejected() as session:
            row = session.get(RawItem, "item001")
            assert row is not None
            row.prefilter_status = PrefilterStatus.REJECTED.value
            row.lang = "en"
            session.commit()
    finally:
        engine.dispose()

    dry = runner.invoke(cli.app, [*base, "extract", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "LLM calls: 0 (dry run)" in dry.output and "[0] (hackernews | Ask HN)" in dry.output

    label = runner.invoke(
        cli.app, [*base, "eval", "label", "--n", "2", "--seed", "1"], input="y\ny\nn\n"
    )
    assert label.exit_code == 0, label.output
    lines = (project / "evals" / "data" / "extraction_gold.jsonl").read_text().splitlines()
    assert len(lines) == 2

    free = runner.invoke(cli.app, [*base, "eval", "extraction"])
    assert free.exit_code == 0, free.output
    assert "Prefilter (free): recall" in free.output and "--yes" in free.output

    backend = FakeBackend([{"problems": [problem(0)]}])
    monkeypatch.setattr(cli, "build_backend", lambda settings: backend)
    paid = runner.invoke(cli.app, [*base, "eval", "extraction", "--yes"])
    assert paid.exit_code == 0, paid.output
    assert "Extraction: precision" in paid.output
    saved = json.loads((project / "evals" / "results" / "extraction_latest.json").read_text())
    assert saved["extraction"]["llm_calls"] == 1

    backend.queue({"problems": [problem(0)]})
    real = runner.invoke(cli.app, [*base, "extract"])
    assert real.exit_code == exit_codes.OK, real.output
    assert "Processed 1 items: 1 problems (1 with verified quotes)" in real.output
    again = runner.invoke(cli.app, [*base, "extract"])
    assert again.exit_code == exit_codes.OK and "Nothing to extract" in again.output
