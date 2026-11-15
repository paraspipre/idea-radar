"""Stage 3-4: clustering (hash embedder), naming (FakeBackend), demand math, graph via CLI."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from idea_radar import cli
from idea_radar.config import Settings, load_settings
from idea_radar.db.models import Cluster, ClusterMember, Problem, RawItem
from idea_radar.db.session import make_engine, make_session_factory
from idea_radar.embed import HashEmbedder
from idea_radar.llm.factory import build_client
from idea_radar.llm.fake import FakeBackend
from idea_radar.pipeline.cluster import name_clusters, run_clustering
from idea_radar.pipeline.demand import run_demand
from idea_radar.scoring.demand import (
    Signal,
    corroboration,
    demand_scores,
    percentile_ranks,
    raw_components,
    weekly_counts,
)

NOW = datetime(2026, 11, 15, tzinfo=UTC)

INVOICE = "freelancers: reconcile invoices with bank payouts manually every month"
LOGOUT = "mobile users: app keeps logging out and losing unsaved drafts"


def seed(factory: sessionmaker[Session], texts: list[tuple[str, str]]) -> None:
    """texts: (source, 'who: problem')"""
    with factory() as session:
        for k, (source, text) in enumerate(texts):
            who, problem = text.split(": ", 1)
            item_id = f"raw{k:03d}"
            session.add(
                RawItem(
                    id=item_id,
                    source=source,
                    source_id=str(k),
                    query="q",
                    url=f"https://example.test/{k}",
                    snippet=text,
                    created_at=NOW - timedelta(days=k % 20),
                    score=k,
                    content_hash=f"h{k}",
                    prefilter_score=2.0,
                    prefilter_status="passed",
                    prefilter_reasons=[],
                )
            )
            session.flush()  # no ORM relationships: insert the parent row first
            session.add(
                Problem(
                    raw_item_id=item_id,
                    extractor_version="t",
                    who=who,
                    problem=problem,
                    urgency=3,
                    wtp="strong" if source == "github" else "none",
                    wtp_strength=1.0 if source == "github" else 0.0,
                    quote=None,
                    quote_verified=False,
                )
            )
        session.commit()


@pytest.fixture
def settings(config_path: Path) -> Settings:
    settings = load_settings(config_path, environ={}, load_env_file=False)
    clustering = settings.clustering.model_copy(
        update={
            "assign_threshold": 0.95,
            "merge_threshold": 0.97,
            "min_pool": 4,
            "min_member_similarity": 0.8,  # hash collisions give unrelated text ~0.6
        }
    )
    return settings.model_copy(update={"clustering": clustering})


CORPUS = [
    ("hackernews", INVOICE),
    ("github", INVOICE + " again"),
    ("appstore", INVOICE + " too"),
    ("appstore", LOGOUT),
    ("appstore", LOGOUT + " again"),
    ("hackernews", "gardeners: tomato leaves turning yellow in autumn rain"),
]


def test_clustering_groups_similar_problems_and_is_incremental(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    seed(session_factory, CORPUS)
    report = run_clustering(settings, session_factory, HashEmbedder())
    assert report.embedded == 6
    assert report.new_clusters == 2
    with session_factory() as session:
        members = {m.problem_id: m.cluster_id for m in session.scalars(select(ClusterMember))}
        problems = {p.id: p.problem for p in session.scalars(select(Problem))}
    groups: dict[int, set[str]] = {}
    for pid, cid in members.items():
        groups.setdefault(cid, set()).add(problems[pid].split()[0])
    assert sorted(sorted(g) for g in groups.values()) == [["app"], ["reconcile"]]

    again = run_clustering(settings, session_factory, HashEmbedder())
    assert (again.embedded, again.new_clusters, again.assigned_existing) == (0, 0, 0)
    assert again.active_clusters == 2


def test_naming_batches_and_skips_named(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    seed(session_factory, CORPUS)
    run_clustering(settings, session_factory, HashEmbedder())
    backend = FakeBackend(
        [
            {
                "clusters": [
                    {"k": 0, "name": "Name A", "summary": "S", "who": "W"},
                    {"k": 1, "name": "Name B", "summary": "S", "who": "W"},
                ]
            }
        ]
    )
    client = build_client(settings, factory=session_factory, run_id=None, backend=backend)
    report = asyncio.run(name_clusters(settings, session_factory, client))
    assert (report.needing, report.named) == (2, 2)
    assert "[1] (" in backend.calls[0].request.prompt
    again = asyncio.run(name_clusters(settings, session_factory, client))
    assert again.needing == 0 and len(backend.calls) == 1
    with session_factory() as session:
        assert {c.name for c in session.scalars(select(Cluster))} == {"Name A", "Name B"}


def test_demand_math() -> None:
    assert corroboration(1) == 0.0 and corroboration(2) == 0.6 and corroboration(5) == 1.0
    assert percentile_ranks([3.0]) == [0.5]
    assert percentile_ranks([1.0, 2.0, 2.0, 3.0]) == [0.0, 0.5, 0.5, 1.0]
    signals = [
        Signal(NOW - timedelta(days=1), "hackernews", 0.9, 1.0),
        Signal(NOW - timedelta(days=40), "github", 0.1, 0.0),
    ]
    raw = raw_components(signals, now=NOW, half_life_days=30, window_days=28)
    assert raw["corroboration"] == 0.6
    assert raw["wtp"] == 0.5
    assert raw["trend"] == 0.0  # one recent, one prior
    assert 0 < raw["recency"] < 1
    results = demand_scores(
        {1: raw, 2: {**raw, "volume": raw["volume"] + 1, "wtp": 1.0}},
        {"volume": 1.0, "wtp": 1.0},
    )
    assert results[2].score > results[1].score
    assert results[2].normalized["volume"] == 1.0
    assert weekly_counts([NOW - timedelta(days=1), NOW - timedelta(days=8)], now=NOW, weeks=3) == [
        0,
        1,
        1,
    ]


def test_run_demand_scores_every_active_cluster(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    seed(session_factory, CORPUS)
    run_clustering(settings, session_factory, HashEmbedder())
    ranked = run_demand(settings, session_factory, run_id=None, now=NOW)
    assert len(ranked) == 2
    top = ranked[0]
    # the invoice cluster spans 3 sources and has a willing payer -> ranks first
    assert top.size == 3 and top.components["raw"]["corroboration"] == 1.0
    assert ranked[0].score >= ranked[1].score


def test_cli_run_graph_end_to_end(monkeypatch: pytest.MonkeyPatch, config_path: Path) -> None:
    runner = CliRunner()
    base = ["--config", str(config_path)]
    assert runner.invoke(cli.app, [*base, "db", "init"]).exit_code == 0
    settings = load_settings(config_path, environ={}, load_env_file=False)
    engine = make_engine(settings.db_url)
    try:
        seed(make_session_factory(engine), CORPUS)
    finally:
        engine.dispose()
    monkeypatch.setenv("RADAR__CLUSTERING__ASSIGN_THRESHOLD", "0.95")
    monkeypatch.setenv("RADAR__CLUSTERING__MERGE_THRESHOLD", "0.97")
    monkeypatch.setenv("RADAR__CLUSTERING__MIN_POOL", "4")
    monkeypatch.setenv("RADAR__CLUSTERING__EMBEDDING_MODEL", "hash-test")
    monkeypatch.setenv("RADAR__CLUSTERING__MIN_MEMBER_SIMILARITY", "0.8")
    backend = FakeBackend(
        [
            {
                "clusters": [
                    {"k": 0, "name": "First", "summary": "S", "who": "W"},
                    {"k": 1, "name": "Second", "summary": "S", "who": "W"},
                ]
            }
        ]
    )
    monkeypatch.setattr(cli, "build_backend", lambda settings: backend)
    monkeypatch.setattr(cli, "_make_embedder", lambda settings: HashEmbedder())
    result = runner.invoke(cli.app, [*base, "run", "--until", "demand", "--skip", "extract"])
    assert result.exit_code == 0, result.output
    assert "Top clusters by demand" in result.output
    assert "LLM calls this run: 1" in result.output  # nothing pending to extract; 1 naming call
    listing = runner.invoke(cli.app, [*base, "clusters", "--members"])
    assert listing.exit_code == 0, listing.output
    assert "First" in listing.output or "Second" in listing.output
    quality = runner.invoke(cli.app, [*base, "eval", "clusters"])
    assert quality.exit_code == 0, quality.output
    assert "clusters: 2" in quality.output
