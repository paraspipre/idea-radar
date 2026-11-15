"""Apple App Store customer reviews via Apple's public RSS/JSON feed. Keeps low ratings only.

Reviewer names are never stored; reviews link to the app's store page.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from idea_radar.ingest.http import PoliteClient
from idea_radar.ingest.types import FetchedItem, FetchResult

FEED = "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"


def _label(entry: Mapping[str, Any], key: str) -> str | None:
    value = entry.get(key)
    if isinstance(value, Mapping):
        label = value.get("label")
        return str(label) if label is not None else None
    return None


class AppStoreSource:
    name = "appstore"

    def __init__(
        self, apps: Mapping[str, int], *, country: str, max_rating: int, max_pages: int
    ) -> None:
        self._apps = dict(apps)
        self._country = country
        self._max_rating = max_rating
        self._max_pages = max_pages

    def queries(self) -> list[str]:
        return list(self._apps)

    def fetch(
        self, client: PoliteClient, query: str, *, since: datetime, limit: int
    ) -> FetchResult:
        app_id = self._apps[query]
        result = FetchResult()
        newest = since
        for page in range(1, self._max_pages + 1):
            data = client.get_json(FEED.format(country=self._country, page=page, app_id=app_id))
            result.requests += 1
            entries = data.get("feed", {}).get("entry") or []
            if isinstance(entries, Mapping):  # a single entry is not wrapped in a list
                entries = [entries]
            reached_old = False
            for entry in entries:
                item = parse_review(entry, app_name=query, app_id=app_id, country=self._country)
                if item is None:
                    continue
                if item.created_at <= since:
                    reached_old = True
                    continue
                newest = max(newest, item.created_at)
                rating = int(_label(entry, "im:rating") or 5)
                if rating <= self._max_rating:
                    result.items.append(item)
            if reached_old or len(entries) == 0 or len(result.items) >= limit:
                break
        result.items = result.items[:limit]
        result.cursor = newest.isoformat()
        return result


def parse_review(
    entry: Mapping[str, Any], *, app_name: str, app_id: int, country: str
) -> FetchedItem | None:
    review_id = _label(entry, "id")
    rating = _label(entry, "im:rating")
    updated = _label(entry, "updated")
    if not review_id or rating is None or not updated:
        return None  # older feeds start with an app-metadata entry that has no rating
    title = _label(entry, "title")
    body = _label(entry, "content") or ""
    votes = int(_label(entry, "im:voteCount") or 0)
    return FetchedItem(
        source="appstore",
        source_id=f"{app_id}:{review_id}",
        url=f"https://apps.apple.com/{country}/app/id{app_id}?see-all=reviews",
        title=title,
        text=f"{title}. {body}" if title else body,
        created_at=datetime.fromisoformat(updated),
        score=votes,
        num_comments=0,
        context=f"{app_name} (App Store, {rating} stars)",
    )
