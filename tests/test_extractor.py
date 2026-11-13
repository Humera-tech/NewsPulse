"""Tests for text extraction from saved HTML."""

from __future__ import annotations

from newspulse.extractor import (
    MAX_EXCERPT_CHARS,
    MIN_TEXT_CHARS,
    ExtractedText,
    choose_text,
    clean_text,
    extract_article,
    extract_main_text,
    extract_published_date,
    prose_chars,
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
    result = extract_article(
        html, snippet="  The city council   approved a new fleet of electric vehicles on Monday. ")
    assert result.text_type == "feed_excerpt"
    assert result.text == "The city council approved a new fleet of electric vehicles on Monday."


def test_nothing_available() -> None:
    """No HTML and no snippet gives text_type none."""
    result = extract_article(None, snippet=None)
    assert result.text_type == "none"
    assert result.text is None


def test_choose_text_with_full_text() -> None:
    """Long provided text (e.g. Guardian bodyText) is used directly."""
    body = "The council said the new buses would start running on the main routes next month. " * 5
    result = choose_text(body, "snippet")
    assert result.text_type == "full_text"


# A real failure case: a TV schedule page extracted as 387 characters of "text".
TV_SCHEDULE = (
    "Live Now\nAll times eastern\nNOW - 12:30 AM\n12:30 AM\n1:00 AM\n1:30 AM\n2:00 AM\n"
    "2:30 AM\nFox Business Channel\nCOPS\n12:30 AM - 1:00 AM\nPaid Programming\n"
    "1:00 AM - 1:30 AM\nPaid Programming\n1:30 AM - 2:00 AM\nPaid Programming\n"
    "2:00 AM - 2:30 AM\nPaid Programming\n2:30 AM - 3:00 AM\nFox News Channel\n"
    "Jesse Watters Primetime\n1:00 AM - 2:00 AM\nHannity\n2:00 AM - 3:00 AM\n"
    "Fox Weather Channel\nFox News Radio"
)


def test_schedule_page_is_not_full_text() -> None:
    """300+ characters without real sentences are not full_text (the Fox Business case)."""
    assert len(TV_SCHEDULE) >= MIN_TEXT_CHARS
    assert prose_chars(TV_SCHEDULE) == 0
    snippet = "A cybersecurity expert explained the failed Microsoft health check on air this week."
    fallback = choose_text(TV_SCHEDULE, snippet)
    assert fallback.text_type == "feed_excerpt"
    assert fallback.text == snippet


def test_junk_text_and_junk_snippet_give_none() -> None:
    """If the snippet has no complete sentence either, there is no usable text."""
    for snippet in (None, "", "Watch live", "Fox Business"):
        result = choose_text(TV_SCHEDULE, snippet)
        assert result.text_type == "none"
        assert result.text is None


def test_short_prose_is_not_full_text() -> None:
    """Real sentences under 300 characters fall back to the snippet, or are skipped without one."""
    short = "The council approved the plan on Monday after a long debate in the chamber."
    snippet = "Councillors backed the transport plan after a debate lasting several hours."
    assert choose_text(short, snippet) == ExtractedText(snippet, "feed_excerpt")
    assert choose_text(short, None).text_type == "none"


def test_full_text_needs_300_chars_of_prose_not_just_300_chars() -> None:
    """Navigation lines do not count toward the minimum; sentences do."""
    menu = "\n".join(["Home", "Business", "Sport", "Weather", "Subscribe"] * 15)
    prose = "The council said the new buses would start running on the main routes next month. "
    assert len(menu + "\n" + prose) >= MIN_TEXT_CHARS
    assert choose_text(menu + "\n" + prose, None).text_type == "none"
    assert choose_text(menu + "\n" + prose * 4, None).text_type == "full_text"


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
