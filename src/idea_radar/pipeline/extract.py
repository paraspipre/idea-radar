"""Stage 2: batched LLM problem extraction.

Token efficiency:
- only prefilter-passed items, highest prefilter score first;
- ~25 items per call, each a <=500-char snippet; the model lists only the problems it finds
  (non-problems are omitted, not echoed);
- every processed item is recorded per extractor version, so nothing is sent twice;
- on a schema failure the batch is split in half once, then given up on (marked ``error``).

Evidence integrity: a quote is stored only if it appears verbatim in the snippet.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.config import Settings
from idea_radar.db.models import (
    Extraction,
    ExtractionStatus,
    PrefilterStatus,
    Problem,
    RawItem,
)
from idea_radar.llm.client import LLMClient
from idea_radar.llm.errors import LLMOutputError, StopRun
from idea_radar.llm.prompts import Prompt, load_prompt
from idea_radar.llm.types import LLMRequest
from idea_radar.log import get_logger

_log = get_logger(component="extract")

STAGE = "extract"
WTP_STRENGTH = {"none": 0.0, "weak": 0.5, "strong": 1.0}


class ExtractedProblem(BaseModel):
    i: int = Field(ge=0)
    who: str
    problem: str
    workaround: str | None = None
    urgency: int = Field(ge=1, le=5)
    wtp: Literal["none", "weak", "strong"]
    quote: str


class ExtractionBatch(BaseModel):
    problems: list[ExtractedProblem]


@dataclass(frozen=True)
class LLMItem:
    """What the model sees for one item."""

    id: str
    source: str
    snippet: str
    title: str | None = None
    context: str | None = None


@dataclass
class ExtractOutcome:
    problems: dict[str, ExtractedProblem] = field(default_factory=dict)
    not_problems: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def extractor_version(settings: Settings) -> str:
    return f"extract.v{settings.extraction.prompt_version}:{settings.llm.models.small}"


def render_items(items: Sequence[LLMItem]) -> str:
    lines = ["Items:"]
    for index, item in enumerate(items):
        meta = item.source + (f" | {item.context}" if item.context else "")
        text = f"{item.title}. {item.snippet}" if item.title else item.snippet
        lines.append(f"[{index}] ({meta}) {text}")
    return "\n".join(lines)


def build_request(
    items: Sequence[LLMItem], *, prompt: Prompt, settings: Settings
) -> LLMRequest[ExtractionBatch]:
    return LLMRequest(
        stage=STAGE,
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        system=prompt.text,
        prompt=render_items(items),
        schema=ExtractionBatch,
        tier="small",
        max_turns=settings.extraction.max_turns,
        n_items=len(items),
    )


_NORM = re.compile(r"[\W_]+")


def quote_in(quote: str, item: LLMItem) -> bool:
    """True if the quote (ignoring case/punctuation/ellipses) appears in the item's text."""
    needle = _NORM.sub(" ", quote.lower()).strip()
    haystack = _NORM.sub(" ", f"{item.title or ''} {item.snippet}".lower())
    return len(needle) >= 8 and needle in haystack


async def extract_items(
    client: LLMClient, items: Sequence[LLMItem], *, prompt: Prompt, settings: Settings
) -> ExtractOutcome:
    """Extract problems from ``items`` (one call per batch). Raises StopRun to stop the run."""
    outcome = ExtractOutcome()
    size = settings.extraction.batch_size
    for start in range(0, len(items), size):
        await _extract_batch(client, list(items[start : start + size]), prompt, settings, outcome)
    return outcome


async def _extract_batch(
    client: LLMClient,
    batch: list[LLMItem],
    prompt: Prompt,
    settings: Settings,
    outcome: ExtractOutcome,
    *,
    may_split: bool = True,
) -> None:
    try:
        result = await client.complete_json(build_request(batch, prompt=prompt, settings=settings))
    except LLMOutputError as exc:
        if may_split and len(batch) > 1:
            _log.warning("extract_batch_split", size=len(batch), error=str(exc))
            half = len(batch) // 2
            for part in (batch[:half], batch[half:]):
                await _extract_batch(client, part, prompt, settings, outcome, may_split=False)
            return
        _log.error("extract_batch_failed", size=len(batch), error=str(exc))
        outcome.failed.extend(item.id for item in batch)
        return
    found: set[str] = set()
    for problem in result.data.problems:
        if problem.i >= len(batch):
            _log.warning("extract_bad_index", index=problem.i, size=len(batch))
            continue
        item = batch[problem.i]
        if item.id in found:
            continue  # one problem per item; the first is kept
        found.add(item.id)
        outcome.problems[item.id] = problem
    outcome.not_problems.extend(item.id for item in batch if item.id not in found)


def pending_items(
    factory: sessionmaker[Session], *, version: str, limit: int | None = None
) -> list[LLMItem]:
    processed = select(Extraction.raw_item_id).where(Extraction.extractor_version == version)
    stmt = (
        select(RawItem)
        .where(
            RawItem.prefilter_status == PrefilterStatus.PASSED.value,
            RawItem.id.not_in(processed),
        )
        .order_by(RawItem.prefilter_score.desc(), RawItem.created_at.desc())
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    with factory() as session:
        return [
            LLMItem(
                id=row.id,
                source=row.source,
                snippet=row.snippet,
                title=row.title,
                context=row.context,
            )
            for row in session.scalars(stmt)
        ]


def save_outcome(
    factory: sessionmaker[Session],
    items: Sequence[LLMItem],
    outcome: ExtractOutcome,
    *,
    version: str,
    run_id: str | None,
) -> None:
    by_id = {item.id: item for item in items}
    with factory() as session:
        for item_id, extracted in outcome.problems.items():
            verified = quote_in(extracted.quote, by_id[item_id])
            session.add(
                Problem(
                    raw_item_id=item_id,
                    extractor_version=version,
                    who=extracted.who.strip(),
                    problem=extracted.problem.strip(),
                    workaround=(extracted.workaround or "").strip() or None,
                    urgency=extracted.urgency,
                    wtp=extracted.wtp,
                    wtp_strength=WTP_STRENGTH[extracted.wtp],
                    quote=extracted.quote.strip() if verified else None,
                    quote_verified=verified,
                )
            )
        statuses = (
            [(i, ExtractionStatus.PROBLEM) for i in outcome.problems]
            + [(i, ExtractionStatus.NOT_PROBLEM) for i in outcome.not_problems]
            + [(i, ExtractionStatus.ERROR) for i in outcome.failed]
        )
        for item_id, status in statuses:
            session.merge(
                Extraction(
                    raw_item_id=item_id,
                    extractor_version=version,
                    status=status.value,
                    run_id=run_id,
                )
            )
        session.commit()


@dataclass
class ExtractionReport:
    pending: int = 0
    processed: int = 0
    problems: int = 0
    not_problems: int = 0
    failed: int = 0
    quotes_verified: int = 0
    stopped: str | None = None
    stop: StopRun | None = None


async def run_extraction(
    settings: Settings,
    factory: sessionmaker[Session],
    client: LLMClient,
    *,
    run_id: str | None,
    max_batches: int | None = None,
) -> ExtractionReport:
    """Process pending items batch by batch, saving after each batch (so a stop loses nothing)."""
    version = extractor_version(settings)
    prompt = load_prompt("extract", settings.extraction.prompt_version)
    size = settings.extraction.batch_size
    limit = max_batches * size if max_batches is not None else None
    items = pending_items(factory, version=version, limit=limit)
    report = ExtractionReport(pending=len(items))
    for start in range(0, len(items), size):
        batch = items[start : start + size]
        try:
            outcome = await extract_items(client, batch, prompt=prompt, settings=settings)
        except StopRun as exc:
            report.stopped = str(exc)
            report.stop = exc
            break
        save_outcome(factory, batch, outcome, version=version, run_id=run_id)
        by_id = {item.id: item for item in batch}
        report.processed += len(batch)
        report.problems += len(outcome.problems)
        report.not_problems += len(outcome.not_problems)
        report.failed += len(outcome.failed)
        report.quotes_verified += sum(
            1 for item_id, p in outcome.problems.items() if quote_in(p.quote, by_id[item_id])
        )
    return report
