"""Versioned system prompts: `<name>.v<version>.md`. Bump the version to change a prompt;
the version is part of the cache key, so old outputs are never mixed with new ones."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    text: str


def load_prompt(name: str, version: int) -> Prompt:
    resource = files(__name__).joinpath(f"{name}.v{version}.md")
    if not resource.is_file():
        raise FileNotFoundError(f"Prompt {name}.v{version}.md not found in {__name__}")
    return Prompt(name=name, version=version, text=resource.read_text(encoding="utf-8").strip())
