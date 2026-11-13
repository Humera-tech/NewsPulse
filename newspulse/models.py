"""Pydantic data models and UTC timestamp helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DiscoveredVia = Literal["gdelt", "guardian", "gnews", "google_news"]
TextType = Literal["full_text", "feed_excerpt", "none"]
SummaryMethod = Literal["gemini", "extractive"]

TRACKED_FIELDS: tuple[str, ...] = ("title", "source", "url", "published_at", "excerpt")


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def to_iso_utc(dt: datetime) -> str:
    """Format a datetime as ISO 8601 UTC with a trailing Z, e.g. 2026-11-12T10:30:00Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return to_iso_utc(utc_now())


def parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO 8601 string (with Z, offset or date only) into a UTC datetime."""
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def shorten_message(text: str | None, limit: int = 120) -> str:
    """Return the first line of a message, whitespace-collapsed, cut at a word boundary."""
    first_line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    message = " ".join(first_line.split())
    if len(message) <= limit:
        return message
    cut = message[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-(")
    return f"{cut or message[:limit]}..."


def _is_empty(value: object) -> bool:
    """Return True for None, empty strings and whitespace-only strings."""
    return value is None or (isinstance(value, str) and not value.strip())


class Article(BaseModel):
    """One collected news article with its text and summary."""

    title: str | None = None
    source: str | None = None
    url: str | None = None
    published_at: str | None = None
    retrieved_at: str
    discovered_via: DiscoveredVia
    excerpt: str | None = None
    text_type: TextType = "none"
    summary: str | None = None
    summary_method: SummaryMethod | None = None
    missing_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_missing(self) -> "Article":
        """Populate missing_fields automatically on creation."""
        fill_missing_fields(self)
        return self


def fill_missing_fields(article: Article) -> Article:
    """Recompute article.missing_fields from its current values and return it.

    Tracks title/source/url/published_at/excerpt, plus "summary" when the
    article has no summary and was already flagged for it.
    """
    missing = [name for name in TRACKED_FIELDS if _is_empty(getattr(article, name))]
    if "summary" in article.missing_fields and _is_empty(article.summary):
        missing.append("summary")
    article.missing_fields = missing
    return article


class SourceError(BaseModel):
    """A source that failed during a run, and why."""

    source: str
    error: str


class RunResult(BaseModel):
    """Everything produced by one pipeline run."""

    topic: str
    generated_at: str
    days: int
    requested_limit: int
    article_count: int
    summarizer_requested: str | None = None
    sources_queried: list[str] = Field(default_factory=list)
    articles_by_source: dict[str, int] = Field(default_factory=dict)
    source_errors: list[SourceError] = Field(default_factory=list)
    shortfall_note: str | None = None
    articles: list[Article] = Field(default_factory=list)
