"""Source adapters against httpx.MockTransport, using the real response shapes (probed live)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from idea_radar.config import HNQuery
from idea_radar.ingest.http import PoliteClient
from idea_radar.ingest.sources.appstore import AppStoreSource
from idea_radar.ingest.sources.github import GitHubSource
from idea_radar.ingest.sources.hackernews import HackerNewsSource
from idea_radar.ingest.sources.stackexchange import StackExchangeSource

SINCE = datetime(2026, 10, 1, tzinfo=UTC)


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> PoliteClient:
    return PoliteClient(
        user_agent="test",
        timeout_s=5,
        min_interval_s=0,
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )


def test_hackernews_parses_comments_and_sets_cursor() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "objectID": "50307576",
                        "created_at_i": 1793382579,
                        "comment_text": "<p>Is there a tool in Windows that dumps a process?</p>",
                        "story_title": "Why a program doesn't work from a batch file",
                        "author": "someone",
                        "points": None,
                    }
                ],
                "nbPages": 1,
            },
        )

    source = HackerNewsSource([HNQuery(query="is there a tool", tags="comment")])
    (query,) = source.queries()
    result = source.fetch(client_for(handler), query, since=SINCE, limit=50)
    assert len(result.items) == 1
    item = result.items[0]
    assert item.url == "https://news.ycombinator.com/item?id=50307576"
    assert item.context == "Why a program doesn't work from a batch file"
    assert result.cursor == "1793382579"
    assert seen[0].url.params["numericFilters"] == f"created_at_i>{int(SINCE.timestamp())}"
    assert "someone" not in repr(item)  # author is never kept


def test_stackexchange_pages_and_respects_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    pauses: list[float] = []
    pages: list[dict[str, Any]] = [
        {
            "items": [
                {
                    "question_id": 95787,
                    "creation_date": 1794574186,
                    "link": "https://softwarerecs.stackexchange.com/questions/95787/x",
                    "title": "What software can manage large volumes of data?",
                    "body": "<p>I'm researching ways to manage data.</p>",
                    "score": 2,
                    "view_count": 100,
                    "answer_count": 1,
                    "tags": ["cloud-storage"],
                    "owner": {"display_name": "someone"},
                }
            ],
            "has_more": True,
            "backoff": 3,
            "quota_remaining": 250,
        },
        {"items": [], "has_more": False, "quota_remaining": 249},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pages[int(request.url.params["page"]) - 1])

    client = client_for(handler)
    monkeypatch.setattr(client, "pause", pauses.append)
    source = StackExchangeSource(["softwarerecs"], [""], key=None)
    result = source.fetch(client, "softwarerecs:", since=SINCE, limit=50)
    assert [i.source_id for i in result.items] == ["softwarerecs:95787"]
    assert result.items[0].score == 4  # score + views // 50
    assert pauses == [3.0]
    assert result.requests == 2
    assert result.cursor == "1794574186"


def test_github_skips_pull_requests_and_uses_reactions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "created:>2026-10-01T00:00:00Z" in request.url.params["q"]
        assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": 1,
                        "html_url": "https://github.com/o/r/issues/1",
                        "repository_url": "https://api.github.com/repos/o/r",
                        "title": "[Feature] Export to CSV",
                        "body": "Please add CSV export, doing it manually is painful.",
                        "comments": 4,
                        "created_at": "2026-10-27T18:55:06Z",
                        "reactions": {"total_count": 42},
                    },
                    {"id": 2, "pull_request": {}, "created_at": "2026-10-28T00:00:00Z"},
                ]
            },
        )

    source = GitHubSource(["is:issue label:enhancement"], token=None)
    result = source.fetch(client_for(handler), "is:issue label:enhancement", since=SINCE, limit=50)
    assert len(result.items) == 1
    assert result.items[0].score == 42
    assert result.items[0].context == "o/r"
    assert result.cursor is not None and result.cursor.startswith("2026-10-27T18:55:06")


def _review(review_id: str, rating: int, updated: str) -> dict[str, Any]:
    return {
        "id": {"label": review_id},
        "im:rating": {"label": str(rating)},
        "updated": {"label": updated},
        "title": {"label": "Keeps logging me out"},
        "content": {"label": "Every update loses my drafts.", "attributes": {"type": "text"}},
        "author": {"name": {"label": "Reviewer Name"}},
        "im:voteCount": {"label": "3"},
    }


def test_appstore_keeps_low_ratings_and_stops_at_cursor() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        entries = [
            {"im:name": {"label": "App metadata entry"}},  # no rating -> skipped
            _review("1", 1, "2026-11-13T00:54:49-07:00"),
            _review("2", 5, "2026-11-12T00:00:00-07:00"),  # high rating -> skipped
            _review("3", 2, "2026-09-01T00:00:00-07:00"),  # older than cursor -> stop
        ]
        return httpx.Response(200, json={"feed": {"entry": entries}})

    source = AppStoreSource({"Slack": 618783545}, country="us", max_rating=2, max_pages=3)
    result = source.fetch(client_for(handler), "Slack", since=SINCE, limit=50)
    assert [i.source_id for i in result.items] == ["618783545:1"]
    item = result.items[0]
    assert item.score == 3
    assert item.context == "Slack (App Store, 1 stars)"
    assert "Reviewer Name" not in repr(item)
    assert len(calls) == 1  # stopped paging once it reached already-seen reviews
    assert result.cursor is not None and result.cursor.startswith("2026-11-13")


def test_client_retries_429_then_raises_on_404() -> None:
    statuses = iter([429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, json={"ok": True}, headers={"Retry-After": "1"})

    assert client_for(handler).get_json("https://example.test/x") == {"ok": True}

    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    try:
        client_for(not_found).get_json("https://example.test/x")
    except Exception as exc:
        assert "404" in str(exc)
    else:
        raise AssertionError("expected SourceError")
