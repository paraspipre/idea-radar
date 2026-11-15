"""Hacker News via the Algolia HN Search API (public, no auth)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from idea_radar.config import HNQuery
from idea_radar.ingest.http import PoliteClient
from idea_radar.ingest.types import FetchedItem, FetchResult

API = "https://hn.algolia.com/api/v1/search_by_date"
PAGE_SIZE = 100


class HackerNewsSource:
    name = "hackernews"

    def __init__(self, queries: Sequence[HNQuery]) -> None:
        self._queries = {f"{q.tags}:{q.query}": q for q in queries}

    def queries(self) -> list[str]:
        return list(self._queries)

    def fetch(
        self, client: PoliteClient, query: str, *, since: datetime, limit: int
    ) -> FetchResult:
        spec = self._queries[query]
        result = FetchResult()
        newest = int(since.timestamp())
        page = 0
        while len(result.items) < limit:
            data = client.get_json(
                API,
                params={
                    "query": spec.query,
                    "tags": spec.tags,
                    "numericFilters": f"created_at_i>{int(since.timestamp())}",
                    "hitsPerPage": PAGE_SIZE,
                    "page": page,
                },
            )
            result.requests += 1
            hits: list[dict[str, Any]] = data.get("hits", [])
            for hit in hits:
                item = parse_hit(hit)
                if item is not None:
                    result.items.append(item)
                    newest = max(newest, int(item.created_at.timestamp()))
            page += 1
            if not hits or page >= int(data.get("nbPages", 0)):
                break
        result.items = result.items[:limit]
        result.cursor = str(newest)
        return result


def parse_hit(hit: dict[str, Any]) -> FetchedItem | None:
    object_id = hit.get("objectID")
    created = hit.get("created_at_i")
    text = hit.get("comment_text") or hit.get("story_text") or ""
    title = hit.get("title")
    if not object_id or created is None or not (text or title):
        return None
    return FetchedItem(
        source="hackernews",
        source_id=str(object_id),
        url=f"https://news.ycombinator.com/item?id={object_id}",
        title=title,
        text=text if text else (title or ""),
        created_at=datetime.fromtimestamp(int(created), UTC),
        score=int(hit.get("points") or 0),
        num_comments=int(hit.get("num_comments") or 0),
        context=hit.get("story_title") or None,
    )
