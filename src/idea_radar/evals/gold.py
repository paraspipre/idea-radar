"""Gold labels for extraction: one JSON object per line, self-contained (snippet included).

Snippets are the same short, PII-scrubbed text the pipeline stores, with a link back, so the
eval is reproducible even after raw items are purged.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.clock import utc_now
from idea_radar.config import Settings
from idea_radar.db.models import PrefilterStatus, RawItem

GOLD_FILE = Path("evals") / "data" / "extraction_gold.jsonl"


class GoldItem(BaseModel):
    id: str
    source: str
    url: str
    title: str | None = None
    context: str | None = None
    snippet: str
    prefilter_status: str
    prefilter_score: float
    is_problem: bool
    wtp: bool = False
    note: str | None = None
    labeled_at: datetime


def gold_path(settings: Settings) -> Path:
    return settings.root / GOLD_FILE


def load_gold(path: Path) -> list[GoldItem]:
    if not path.is_file():
        return []
    items: list[GoldItem] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                items.append(GoldItem.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid gold item: {exc}") from exc
    return items


def append_gold(path: Path, item: GoldItem) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")


def candidates_to_label(
    factory: sessionmaker[Session], *, exclude: set[str], n: int, seed: int | None = None
) -> list[RawItem]:
    """A shuffled mix: half prefilter-passed, half rejected English items.

    Including rejected items is what makes prefilter recall measurable.
    """
    rng = random.Random(seed)
    with factory() as session:
        passed = list(
            session.scalars(
                select(RawItem).where(RawItem.prefilter_status == PrefilterStatus.PASSED.value)
            )
        )
        rejected = list(
            session.scalars(
                select(RawItem).where(
                    RawItem.prefilter_status == PrefilterStatus.REJECTED.value,
                    RawItem.lang == "en",
                )
            )
        )
    passed = [r for r in passed if r.id not in exclude]
    rejected = [r for r in rejected if r.id not in exclude]
    rng.shuffle(passed)
    rng.shuffle(rejected)
    half = n // 2
    picked = passed[:half] + rejected[: n - min(half, len(passed))]
    rng.shuffle(picked)
    return picked[:n]


def gold_from_row(row: RawItem, *, is_problem: bool, wtp: bool, note: str | None) -> GoldItem:
    return GoldItem(
        id=row.id,
        source=row.source,
        url=row.url,
        title=row.title,
        context=row.context,
        snippet=row.snippet,
        prefilter_status=row.prefilter_status,
        prefilter_score=row.prefilter_score,
        is_problem=is_problem,
        wtp=wtp,
        note=note,
        labeled_at=utc_now(),
    )
