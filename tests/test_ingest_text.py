from __future__ import annotations

from idea_radar.ingest.prefilter import engagement_bonus, evaluate
from idea_radar.ingest.text import (
    content_hash,
    html_to_text,
    is_probably_english,
    scrub_pii,
    snippet_around,
)

PRIOR = {"appstore": 1.0}


def check(text: str, **kwargs: object) -> tuple[float, bool, list[str]]:
    params: dict[str, object] = {
        "source": "hackernews",
        "score": 0,
        "num_comments": 0,
        "threshold": 1.5,
        "min_chars": 40,
        "source_prior": PRIOR,
    }
    params.update(kwargs)
    result = evaluate(text, **params)  # type: ignore[arg-type]
    return result.score, result.passed, result.reasons


def test_html_to_text_strips_tags_and_entities() -> None:
    raw = "<p>Is there a tool&#x27;s <i>alternative</i>?</p><p>Second&nbsp;para &amp; more</p>"
    assert html_to_text(raw) == "Is there a tool's alternative? Second para & more"


def test_scrub_pii() -> None:
    text = (
        "Mail me at jane.doe+x@example.com or call +1 (415) 555-0100, ping @jdoe_99, "
        "see https://example.com/page?token=abc123&user=7"
    )
    scrubbed = scrub_pii(text)
    assert "example.com/page" in scrubbed
    for secret in ("jane.doe", "555-0100", "@jdoe_99", "token=abc123"):
        assert secret not in scrubbed
    assert "[email]" in scrubbed and "[phone]" in scrubbed and "@user" in scrubbed


def test_english_heuristic() -> None:
    assert is_probably_english("Is there a tool that can sync my calendar with the team board?")
    assert not is_probably_english("Gibt es ein Werkzeug, das meinen Kalender synchronisiert?")


def test_snippet_window_is_bounded_and_centered() -> None:
    text = ("filler words here " * 60) + "I wish there was a tool for invoices. " + ("tail " * 80)
    position = text.index("I wish")
    snippet = snippet_around(text, position, 200)
    assert len(snippet) <= 206  # 200 + ellipses
    assert "I wish there was a tool" in snippet
    assert snippet.startswith("...") and snippet.endswith("...")
    assert snippet_around("short text", 3, 200) == "short text"


def test_content_hash_ignores_case_and_punctuation() -> None:
    assert content_hash("Hello, World!") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello there")


def test_problem_and_wtp_signals_pass() -> None:
    score, passed, reasons = check(
        "I wish there was a tool to reconcile Stripe payouts. I'd happily pay $20/month for it."
    )
    assert passed
    assert "problem:wish" in reasons
    assert "wtp:would_pay" in reasons and "wtp:price_point" in reasons
    assert score >= 3.0


def test_plain_chatter_is_rejected() -> None:
    _, passed, _ = check("This is a great article about the history of compilers and parsers.")
    assert not passed


def test_negative_signals_cost_points() -> None:
    _, passed, reasons = check("We're hiring! Is there a tool you love? Apply now for this role.")
    assert "negative:hiring" in reasons
    assert not passed


def test_short_and_non_english_are_rejected_with_reasons() -> None:
    assert check("too short")[2] == ["too_short"]
    assert check("Gibt es ein Werkzeug, das meinen Kalender mit dem Team synchronisiert?")[2] == [
        "not_english"
    ]


def test_source_prior_and_engagement() -> None:
    # One pain signal (1.0): not enough alone, but a 1-star review's prior (1.0) tips it over.
    text = "The new update keeps logging me out and loses my drafts. So frustrating."
    _, passed_hn, _ = check(text, source="hackernews")
    _, passed_review, reasons = check(text, source="appstore")
    assert not passed_hn
    assert passed_review and "source_prior:1" in reasons
    assert engagement_bonus(0, 0) == 0.0
    assert engagement_bonus(1000, 1000) == 1.0
