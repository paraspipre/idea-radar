"""Source adapters (official APIs only) and the registry that builds the enabled ones."""

from __future__ import annotations

import os
from collections.abc import Mapping

from idea_radar.config import Settings
from idea_radar.ingest.sources.appstore import AppStoreSource
from idea_radar.ingest.sources.github import GitHubSource
from idea_radar.ingest.sources.hackernews import HackerNewsSource
from idea_radar.ingest.sources.stackexchange import StackExchangeSource
from idea_radar.ingest.types import Source

ALL_SOURCES = ("hackernews", "stackexchange", "github", "appstore", "producthunt", "reddit")
TOKEN_GATED = {"producthunt": "PRODUCTHUNT_TOKEN", "reddit": "REDDIT_CLIENT_ID"}


def build_sources(
    settings: Settings,
    *,
    only: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[list[tuple[Source, float]], list[str]]:
    """Return ([(source, min_interval_s)], [skip notes]) for enabled sources."""
    env = os.environ if environ is None else environ
    cfg = settings.sources
    wanted = set(only) if only else set(ALL_SOURCES)
    sources: list[tuple[Source, float]] = []
    notes: list[str] = []

    if "hackernews" in wanted and cfg.hackernews.enabled:
        sources.append((HackerNewsSource(cfg.hackernews.queries), cfg.hackernews.min_interval_s))
    if "stackexchange" in wanted and cfg.stackexchange.enabled:
        key = env.get("STACKEXCHANGE_KEY", "").strip() or None
        se = StackExchangeSource(cfg.stackexchange.sites, cfg.stackexchange.queries, key=key)
        sources.append((se, cfg.stackexchange.min_interval_s))
    if "github" in wanted and cfg.github.enabled:
        token = env.get("GITHUB_TOKEN", "").strip() or None
        interval = (
            cfg.github.min_interval_s if token is None else min(cfg.github.min_interval_s, 2.5)
        )
        sources.append((GitHubSource(cfg.github.queries, token=token), interval))
    if "appstore" in wanted and cfg.appstore.enabled:
        app = cfg.appstore
        store = AppStoreSource(
            app.apps, country=app.country, max_rating=app.max_rating, max_pages=app.max_pages
        )
        sources.append((store, app.min_interval_s))
    for name, env_var in TOKEN_GATED.items():
        if name in wanted:
            enabled = getattr(cfg, name).enabled
            has_token = bool(env.get(env_var, "").strip())
            if enabled or has_token:
                notes.append(f"{name}: adapter not built yet (needs {env_var}); skipped")
    return sources, notes
