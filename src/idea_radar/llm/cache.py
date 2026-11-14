"""Content-hash cache: identical requests are never sent twice."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from idea_radar.db.models import LLMCacheEntry
from idea_radar.llm.types import EffortLevel, LLMRequest


def cache_key(*, backend: str, model: str, effort: EffortLevel, request: LLMRequest[Any]) -> str:
    """sha256 over everything that can change the output. Bump prompt_version to invalidate."""
    payload = {
        "backend": backend,
        "model": model,
        "effort": effort,
        "stage": request.stage,
        "prompt_name": request.prompt_name,
        "prompt_version": request.prompt_version,
        "system": request.system,
        "prompt": request.prompt,
        "schema": request.schema.model_json_schema(),
        "tools": list(request.tools),
        "max_turns": request.max_turns,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def get(self, key: str) -> dict[str, Any] | None:
        with self._factory() as session:
            entry = session.get(LLMCacheEntry, key)
            return dict(entry.response) if entry is not None else None

    def put(
        self,
        key: str,
        *,
        stage: str,
        model: str,
        prompt_name: str,
        prompt_version: int,
        response: dict[str, Any],
    ) -> None:
        with self._factory() as session:
            session.merge(
                LLMCacheEntry(
                    key=key,
                    stage=stage,
                    model=model,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    response=response,
                )
            )
            session.commit()
