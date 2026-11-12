"""Main-text and date extraction from article HTML using trafilatura."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import trafilatura

from newspulse.models import parse_iso, to_iso_utc

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 300
MAX_EXCERPT_CHARS = 5000
_SPACES_RE = re.compile(r"[ \t\r\f\v ]+")
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]?(?=\s)")


@dataclass(frozen=True)
class ExtractedText:
    """Result of choosing the best available text for an article."""

    text: str | None
    text_type: str  # "full_text" | "feed_excerpt" | "none"
    published_at: str | None = None


def clean_text(text: str | None) -> str:
    """Collapse runs of spaces and blank lines; keep single line breaks between paragraphs."""
    if not text:
        return ""
    lines = (_SPACES_RE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def truncate_text(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    """Shorten text to at most `limit` characters, cutting at a sentence end when possible."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    ends = [m.end() for m in _SENTENCE_END_RE.finditer(window + " ")]
    cut = ends[-1] if ends and ends[-1] > limit // 2 else window.rfind(" ")
    return window[: cut if cut > 0 else limit].rstrip()


def extract_main_text(html: str | None) -> str | None:
    """Return the cleaned main article text from HTML, or None if extraction fails."""
    if not html:
        return None
    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
    except Exception as exc:  # noqa: BLE001 - malformed pages must not crash the run
        logger.debug("trafilatura.extract failed: %s", exc)
        return None
    return clean_text(text) or None


def extract_published_date(html: str | None) -> str | None:
    """Return the page's publication date (ISO 8601 UTC) from its metadata, if found."""
    if not html:
        return None
    try:
        metadata = trafilatura.extract_metadata(html)
    except Exception as exc:  # noqa: BLE001
        logger.debug("trafilatura.extract_metadata failed: %s", exc)
        return None
    parsed = parse_iso(getattr(metadata, "date", None)) if metadata else None
    return to_iso_utc(parsed) if parsed else None


def choose_text(main_text: str | None, snippet: str | None) -> ExtractedText:
    """Pick full text if long enough, else the feed snippet, else nothing."""
    text = clean_text(main_text)
    if len(text) >= MIN_TEXT_CHARS:
        return ExtractedText(truncate_text(text), "full_text")
    fallback = clean_text(snippet)
    if fallback:
        return ExtractedText(truncate_text(fallback), "feed_excerpt")
    return ExtractedText(None, "none")


def extract_article(html: str | None, snippet: str | None, need_date: bool = False) -> ExtractedText:
    """Extract text (and optionally the date) from a fetched page, with snippet fallback."""
    chosen = choose_text(extract_main_text(html), snippet)
    if not need_date:
        return chosen
    return ExtractedText(chosen.text, chosen.text_type, extract_published_date(html))
