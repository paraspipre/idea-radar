"""Value types produced by source adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from idea_radar.ingest.http import PoliteClient


@dataclass(frozen=True)
class FetchedItem:
    """One item as a source returns it, before cleaning. ``text`` may contain HTML."""

    source: str
    source_id: str
    url: str
    title: str | None
    text: str
    created_at: datetime
    score: int = 0
    num_comments: int = 0
    context: str | None = None  # e.g. parent story title, repo name, app name

    @property
    def item_id(self) -> str:
        return hashlib.sha256(f"{self.source}:{self.source_id}".encode()).hexdigest()[:32]


@dataclass
class FetchResult:
    items: list[FetchedItem] = field(default_factory=list)
    cursor: str | None = None  # new high-water mark to save, or None to keep the old one
    requests: int = 0


class Source(Protocol):
    """A source adapter. ``queries`` are the cursor keys; each is fetched independently."""

    name: str

    def queries(self) -> list[str]: ...

    def fetch(
        self, client: PoliteClient, query: str, *, since: datetime, limit: int
    ) -> FetchResult: ...
