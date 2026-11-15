"""Persistence for ingest: cursors, idempotent item upserts with dedupe, retention purge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.clock import utc_now
from idea_radar.db.models import IngestCursor, PrefilterStatus, RawItem


@dataclass(frozen=True)
class PreparedItem:
    """A cleaned, scrubbed, prefiltered item ready to store."""

    id: str
    source: str
    source_id: str
    query: str
    url: str
    title: str | None
    snippet: str
    context: str | None
    created_at: datetime
    score: int
    num_comments: int
    lang: str
    content_hash: str
    prefilter_score: float
    passed: bool
    reasons: list[str]
    has_wtp: bool


@dataclass
class SaveStats:
    new: int = 0
    updated: int = 0
    passed: int = 0
    rejected: int = 0
    duplicates: int = 0


class IngestStore:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def get_cursor(self, source: str, key: str) -> str | None:
        with self._factory() as session:
            cursor = session.get(IngestCursor, (source, key))
            return cursor.value if cursor is not None else None

    def set_cursor(self, source: str, key: str, value: str) -> None:
        with self._factory() as session:
            cursor = session.get(IngestCursor, (source, key))
            if cursor is None:
                session.add(IngestCursor(source=source, key=key, value=value))
            else:
                cursor.value = value
                cursor.updated_at = utc_now()
            session.commit()

    def save_items(self, items: Sequence[PreparedItem]) -> SaveStats:
        """Insert new items; refresh engagement on known ones; mark content duplicates."""
        stats = SaveStats()
        if not items:
            return stats
        with self._factory() as session:
            ids = [item.id for item in items]
            existing = {
                row.id: row for row in session.scalars(select(RawItem).where(RawItem.id.in_(ids)))
            }
            hashes = {item.content_hash for item in items}
            known_hashes = set(
                session.scalars(
                    select(RawItem.content_hash).where(RawItem.content_hash.in_(hashes))
                )
            )
            now = utc_now()
            for item in items:
                row = existing.get(item.id)
                if row is not None:
                    row.score = max(row.score, item.score)
                    row.num_comments = max(row.num_comments, item.num_comments)
                    row.fetched_at = now
                    stats.updated += 1
                    continue
                if item.content_hash in known_hashes:
                    status = PrefilterStatus.DUPLICATE
                    stats.duplicates += 1
                elif item.passed:
                    status = PrefilterStatus.PASSED
                    stats.passed += 1
                else:
                    status = PrefilterStatus.REJECTED
                    stats.rejected += 1
                known_hashes.add(item.content_hash)
                existing[item.id] = RawItem(
                    id=item.id,
                    source=item.source,
                    source_id=item.source_id,
                    query=item.query[:200],
                    url=item.url,
                    title=item.title,
                    snippet=item.snippet,
                    context=item.context,
                    created_at=item.created_at,
                    fetched_at=now,
                    score=item.score,
                    num_comments=item.num_comments,
                    lang=item.lang,
                    content_hash=item.content_hash,
                    prefilter_score=item.prefilter_score,
                    prefilter_status=status.value,
                    prefilter_reasons=item.reasons,
                    has_wtp=item.has_wtp,
                )
                session.add(existing[item.id])
                stats.new += 1
            session.commit()
        return stats

    def purge(self, *, older_than: datetime) -> int:
        """Delete rejected/duplicate items older than the retention window (passed items stay)."""
        stmt = delete(RawItem).where(
            RawItem.fetched_at < older_than,
            RawItem.prefilter_status.in_(
                [PrefilterStatus.REJECTED.value, PrefilterStatus.DUPLICATE.value]
            ),
        )
        with self._factory() as session:
            result = session.execute(stmt)
            session.commit()
            return int(getattr(result, "rowcount", 0) or 0)

    def counts(self) -> list[tuple[str, str, int]]:
        stmt = (
            select(RawItem.source, RawItem.prefilter_status, func.count())
            .group_by(RawItem.source, RawItem.prefilter_status)
            .order_by(RawItem.source, RawItem.prefilter_status)
        )
        with self._factory() as session:
            return [(source, status, int(n)) for source, status, n in session.execute(stmt)]

    def top_passed(self, limit: int) -> list[RawItem]:
        stmt = (
            select(RawItem)
            .where(RawItem.prefilter_status == PrefilterStatus.PASSED.value)
            .order_by(RawItem.prefilter_score.desc(), RawItem.created_at.desc())
            .limit(limit)
        )
        with self._factory() as session:
            return list(session.scalars(stmt))
