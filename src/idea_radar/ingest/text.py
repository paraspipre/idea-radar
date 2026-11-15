"""Text cleaning: HTML to text, PII scrubbing, a cheap English check, snippet windows."""

from __future__ import annotations

import hashlib
import html
import re

_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAG = re.compile(r"</?(p|br|div|li|ul|ol|pre|blockquote|h\d)[^>]*>", re.IGNORECASE)
_WS = re.compile(r"\s+")

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_HANDLE = re.compile(r"(?<![\w/])@[A-Za-z0-9_][A-Za-z0-9_-]{1,38}")
_URL_QUERY = re.compile(r"(https?://[^\s?#]+)[?#][^\s]*")

_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "it",
        "this",
        "that",
        "i",
        "you",
        "we",
        "they",
        "my",
        "our",
        "your",
        "not",
        "have",
        "has",
        "do",
        "does",
        "can",
        "would",
        "should",
        "there",
        "what",
        "how",
        "why",
        "when",
        "which",
    ]
)


def html_to_text(raw: str) -> str:
    """Strip tags (keeping block boundaries as spaces), unescape entities, collapse whitespace."""
    text = _BLOCK_TAG.sub(" ", raw)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


def scrub_pii(text: str) -> str:
    """Remove emails, phone numbers, @handles, and URL query strings (often carry tokens/ids)."""
    text = _EMAIL.sub("[email]", text)
    text = _URL_QUERY.sub(r"\1", text)
    text = _PHONE.sub("[phone]", text)
    return _HANDLE.sub("@user", text)


def is_probably_english(text: str) -> bool:
    """Cheap heuristic (no model): enough common English stopwords among the words."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 5:
        return bool(words)  # too short to judge: let the other filters decide
    hits = sum(1 for word in words if word in _STOPWORDS)
    return hits / len(words) >= 0.12


def snippet_around(text: str, position: int | None, max_chars: int) -> str:
    """A window of at most ``max_chars`` around ``position`` (start of text if None)."""
    if len(text) <= max_chars:
        return text
    start = 0 if position is None else max(0, position - max_chars // 4)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    window = text[start:end]
    if start > 0:
        window = "..." + window[window.find(" ") + 1 :] if " " in window else "..." + window
    if end < len(text):
        cut = window.rfind(" ")
        window = (window[:cut] if cut > len(window) // 2 else window) + "..."
    return window


def content_hash(text: str) -> str:
    normalized = _WS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
