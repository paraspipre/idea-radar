"""Stack Exchange API 2.3 (official; 300 requests/day without a key, 10k with one).

Content is CC BY-SA: we keep short snippets with links back (attribution).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from idea_radar.ingest.http import PoliteClient
from idea_radar.ingest.types import FetchedItem, FetchResult

API = "https://api.stackexchange.com/2.3/search/advanced"
PAGE_SIZE = 100


class StackExchangeSource:
    name = "stackexchange"

    def __init__(self, sites: Sequence[str], queries: Sequence[str], *, key: str | None) -> None:
        self._pairs = {f"{site}:{q}": (site, q) for site in sites for q in queries}
        self._key = key

    def queries(self) -> list[str]:
        return list(self._pairs)

    def fetch(
        self, client: PoliteClient, query: str, *, since: datetime, limit: int
    ) -> FetchResult:
        site, text_query = self._pairs[query]
        result = FetchResult()
        newest = int(since.timestamp())
        page = 1
        while len(result.items) < limit:
            params: dict[str, Any] = {
                "site": site,
                "order": "desc",
                "sort": "creation",
                "fromdate": int(since.timestamp()) + 1,
                "pagesize": PAGE_SIZE,
                "page": page,
                "filter": "withbody",
            }
            if text_query:
                params["q"] = text_query
            if self._key:
                params["key"] = self._key
            data = client.get_json(API, params=params)
            result.requests += 1
            for raw in data.get("items", []):
                item = parse_question(raw, site)
                if item is not None:
                    result.items.append(item)
                    newest = max(newest, int(item.created_at.timestamp()))
            if data.get("backoff"):
                client.pause(float(data["backoff"]))
            if not data.get("has_more") or int(data.get("quota_remaining", 1)) <= 5:
                break
            page += 1
        result.items = result.items[:limit]
        result.cursor = str(newest)
        return result


def parse_question(raw: dict[str, Any], site: str) -> FetchedItem | None:
    question_id = raw.get("question_id")
    created = raw.get("creation_date")
    if question_id is None or created is None:
        return None
    return FetchedItem(
        source="stackexchange",
        source_id=f"{site}:{question_id}",
        url=str(raw.get("link") or f"https://{site}.stackexchange.com/q/{question_id}"),
        title=raw.get("title"),
        text=str(raw.get("body") or raw.get("title") or ""),
        created_at=datetime.fromtimestamp(int(created), UTC),
        score=int(raw.get("score") or 0) + int(raw.get("view_count") or 0) // 50,
        num_comments=int(raw.get("answer_count") or 0),
        context=f"{site}.stackexchange: " + ", ".join(raw.get("tags") or []),
    )
