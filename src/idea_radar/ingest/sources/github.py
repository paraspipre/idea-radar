"""GitHub issue search (official REST API). Upvoted feature requests are explicit demand."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from idea_radar.ingest.http import PoliteClient
from idea_radar.ingest.types import FetchedItem, FetchResult

API = "https://api.github.com/search/issues"
PAGE_SIZE = 100
MAX_RESULTS = 1000  # GitHub search hard limit


class GitHubSource:
    name = "github"

    def __init__(self, queries: Sequence[str], *, token: str | None) -> None:
        self._queries = list(queries)
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def queries(self) -> list[str]:
        return list(self._queries)

    def fetch(
        self, client: PoliteClient, query: str, *, since: datetime, limit: int
    ) -> FetchResult:
        result = FetchResult()
        newest = since
        since_q = since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        page = 1
        while len(result.items) < limit and page * PAGE_SIZE <= MAX_RESULTS:
            data = client.get_json(
                API,
                params={
                    "q": f"{query} created:>{since_q}",
                    "sort": "created",
                    "order": "desc",
                    "per_page": PAGE_SIZE,
                    "page": page,
                },
                headers=self._headers,
            )
            result.requests += 1
            raw_items: list[dict[str, Any]] = data.get("items", [])
            for raw in raw_items:
                item = parse_issue(raw)
                if item is not None:
                    result.items.append(item)
                    newest = max(newest, item.created_at)
            if len(raw_items) < PAGE_SIZE:
                break
            page += 1
        result.items = result.items[:limit]
        result.cursor = newest.isoformat()
        return result


def parse_issue(raw: dict[str, Any]) -> FetchedItem | None:
    if raw.get("pull_request") is not None:
        return None
    created = raw.get("created_at")
    if not raw.get("id") or not created:
        return None
    repo = str(raw.get("repository_url", "")).removeprefix("https://api.github.com/repos/")
    reactions = raw.get("reactions") or {}
    return FetchedItem(
        source="github",
        source_id=str(raw["id"]),
        url=str(raw.get("html_url", "")),
        title=raw.get("title"),
        text=str(raw.get("body") or raw.get("title") or ""),
        created_at=datetime.fromisoformat(str(created).replace("Z", "+00:00")),
        score=int(reactions.get("total_count") or 0),
        num_comments=int(raw.get("comments") or 0),
        context=repo or None,
    )
