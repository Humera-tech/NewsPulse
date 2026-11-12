"""Google News RSS search feed source (no key), with lazy redirect-link decoding."""

from __future__ import annotations

import calendar
import logging
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlsplit

import feedparser
from googlenewsdecoder import gnewsdecoder

from newspulse.models import to_iso_utc
from newspulse.sources.base import Candidate, NewsSource, SourceUnavailable
from newspulse.sources.guardian import strip_html

logger = logging.getLogger(__name__)

FEED_TEMPLATE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def build_feed_url(topic: str, days: int) -> str:
    """Build the Google News RSS search URL for a topic and day range."""
    query = f"{quote_plus(topic.strip())}+when:{days}d"
    return FEED_TEMPLATE.format(query=query)


def strip_publisher_suffix(title: str, publisher: str | None) -> str:
    """Remove a trailing ' - <Publisher>' from a Google News headline."""
    title = " ".join(title.split())
    if publisher:
        suffix = f" - {publisher}"
        if title.endswith(suffix):
            return title[: -len(suffix)].strip()
    return title


def _entry_date(entry: feedparser.FeedParserDict) -> str | None:
    """Return the entry's publication date as ISO 8601 UTC, if present."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return to_iso_utc(datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc))


def _entry_snippet(entry: feedparser.FeedParserDict, title: str, publisher: str | None) -> str | None:
    """Return the entry description as plain text, or None if it only repeats the headline."""
    text = strip_html(entry.get("summary"))
    if not text:
        return None
    remainder = text.replace(title, "", 1)
    if publisher:
        remainder = remainder.replace(publisher, "", 1)
    return text if len(remainder.strip(" - ")) > 20 else None


def parse_google_news_feed(content: bytes | str) -> list[Candidate]:
    """Parse a Google News RSS document into candidates (links still Google redirects)."""
    feed = feedparser.parse(content)
    if feed.get("bozo") and not feed.get("entries"):
        reason = feed.get("bozo_exception")
        raise SourceUnavailable(f"Google News feed could not be parsed: {reason}")

    candidates: list[Candidate] = []
    for entry in feed.get("entries", []):
        link = (entry.get("link") or "").strip()
        raw_title = entry.get("title") or ""
        if not link or not raw_title.strip():
            continue
        source_info = entry.get("source") or {}
        publisher = (source_info.get("title") or "").strip() or None
        title = strip_publisher_suffix(raw_title, publisher)
        candidates.append(
            Candidate(
                title=title,
                url=link,
                source=publisher,
                published_at=_entry_date(entry),
                snippet=_entry_snippet(entry, title, publisher),
                discovered_via="google_news",
            )
        )
    return candidates


def is_google_news_link(url: str) -> bool:
    """Return True if the URL points at a news.google.com redirect."""
    return urlsplit(url).netloc.lower().endswith("news.google.com")


def decode_google_news_url(url: str, timeout: float = 10.0) -> str:
    """Decode a Google News redirect link to the publisher URL; keep the original on failure."""
    if not is_google_news_link(url):
        return url
    try:
        result = gnewsdecoder(url, interval=None, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - the decoder may fail in many ways
        logger.warning("Could not decode Google News link (%s); keeping Google URL", exc)
        return url
    if isinstance(result, dict) and result.get("success") and result.get("decoded_url"):
        return str(result["decoded_url"])
    message = result.get("message") if isinstance(result, dict) else "unexpected result"
    logger.warning("Could not decode Google News link (%s); keeping Google URL", message)
    return url


class GoogleNewsSource(NewsSource):
    """Searches the Google News RSS feed."""

    name = "google_news"

    def search(self, topic: str, days: int) -> list[Candidate]:
        """Download and parse the Google News RSS search feed for the topic."""
        url = build_feed_url(topic, days)
        response = self.session.get(url, timeout=self.settings.request_timeout)
        if response.status_code >= 400:
            raise SourceUnavailable(f"Google News returned HTTP {response.status_code}")
        return parse_google_news_feed(response.content)

    def resolve_url(self, candidate: Candidate) -> str:
        """Decode the Google redirect link only when the candidate is actually used."""
        return decode_google_news_url(candidate.url, timeout=self.settings.request_timeout)
