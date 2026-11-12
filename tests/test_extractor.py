"""Tests for text extraction from saved HTML."""

from __future__ import annotations

from newspulse.extractor import (
    MAX_EXCERPT_CHARS,
    MIN_TEXT_CHARS,
    choose_text,
    clean_text,
    extract_article,
    extract_main_text,
    extract_published_date,
    truncate_text,
)
from tests.conftest import read_fixture


def test_extracts_main_text_without_boilerplate() -> None:
    """Article paragraphs are kept; navigation, comments and footer are dropped."""
    text = extract_main_text(read_fixture("article.html"))
    assert text is not None
    assert "twelve fast chargers" in text
    assert "Subscribe now" not in text
    assert "Reader comment" not in text
    assert "All rights reserved" not in text


def test_extract_article_full_text() -> None:
    """A long article is marked full_text and stays within the size limit."""
    result = extract_article(read_fixture("article.html"), snippet="short snippet", need_date=True)
    assert result.text_type == "full_text"
    assert result.text is not None
    assert MIN_TEXT_CHARS <= len(result.text) <= MAX_EXCERPT_CHARS
    assert result.published_at == "2026-11-10T00:00:00Z" or result.published_at.startswith("2026-11-10")


def test_extract_published_date_from_metadata() -> None:
    """The publication date comes from the page metadata."""
    date = extract_published_date(read_fixture("article.html"))
    assert date is not None and date.startswith("2026-11-10")


def test_short_page_falls_back_to_snippet() -> None:
    """Pages with too little text use the feed snippet."""
    html = "<html><body><article><p>Too short.</p></article></body></html>"
    result = extract_article(html, snippet="  A feed   snippet about electric vehicles. ")
    assert result.text_type == "feed_excerpt"
    assert result.text == "A feed snippet about electric vehicles."


def test_nothing_available() -> None:
    """No HTML and no snippet gives text_type none."""
    result = extract_article(None, snippet=None)
    assert result.text_type == "none"
    assert result.text is None


def test_choose_text_with_full_text() -> None:
    """Long provided text (e.g. Guardian bodyText) is used directly."""
    body = "Sentence about the topic. " * 20
    result = choose_text(body, "snippet")
    assert result.text_type == "full_text"


def test_clean_text_collapses_whitespace() -> None:
    """Spaces collapse; blank lines are removed but paragraph breaks are kept."""
    assert clean_text("  a \t b \n\n\n c  d  ") == "a b\nc d"
    assert clean_text(None) == ""


def test_truncate_text_cuts_at_sentence_end() -> None:
    """Long text is cut at a sentence boundary and never exceeds the limit."""
    text = "This is one sentence. " * 400
    cut = truncate_text(text.strip(), 5000)
    assert len(cut) <= 5000
    assert cut.endswith(".")
