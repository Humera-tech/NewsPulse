"""GNews API source (free key in GNEWS_API_KEY).

GNews returns only a truncated `content` field, so pages are fetched and
extracted like GDELT and Google News results; `content` is never used as full text.
"""

from __future__ import annotations

from datetime import timedelta

from newspulse.models import parse_iso, to_iso_utc, utc_now
from newspulse.sources.base import Candidate, NewsSource, SourceUnavailable
from newspulse.sources.guardian import strip_html

GNEWS_URL = "https://gnews.io/api/v4/search"
MAX_RESULTS = 10


def build_query(topic: str) -> str:
    """Quote multi-word topics so GNews searches for the exact phrase."""
    cleaned = " ".join(topic.replace('"', " ").split())
    return f'"{cleaned}"' if " " in cleaned else cleaned


def parse_gnews_response(data: dict) -> list[Candidate]:
    """Turn a GNews /search JSON payload into candidates (no full text)."""
    if not isinstance(data, dict) or not isinstance(data.get("articles", []), list):
        raise SourceUnavailable("GNews returned an unexpected JSON structure")

    candidates: list[Candidate] = []
    for item in data.get("articles") or []:
        url = (item.get("url") or "").strip()
        title = " ".join((item.get("title") or "").split())
        if not url or not title:
            continue
        source_info = item.get("source") or {}
        published = parse_iso(item.get("publishedAt"))
        candidates.append(
            Candidate(
                title=title,
                url=url,
                source=(source_info.get("name") or "").strip() or None,
                published_at=to_iso_utc(published) if published else None,
                snippet=strip_html(item.get("description")),
                full_text=None,  # `content` is truncated by GNews; the page is fetched instead
                discovered_via="gnews",
            )
        )
    return candidates


class GNewsSource(NewsSource):
    """Searches the GNews API for recent English articles."""

    name = "gnews"

    def search(self, topic: str, days: int) -> list[Candidate]:
        """Query GNews for the newest articles about the topic within `days` days."""
        api_key = self.settings.gnews_api_key
        if not api_key:
            raise SourceUnavailable("GNEWS_API_KEY not set")
        params = {
            "q": build_query(topic),
            "lang": "en",
            "max": str(MAX_RESULTS),
            "sortby": "publishedAt",
            "from": to_iso_utc(utc_now() - timedelta(days=days)),
            "apikey": api_key,
        }
        response = self.session.get(GNEWS_URL, params=params, timeout=self.settings.request_timeout)
        status = response.status_code
        if status in (401, 403):
            raise SourceUnavailable(f"GNews rejected the API key (HTTP {status})")
        if status == 429:
            raise SourceUnavailable("GNews daily request limit reached (HTTP 429)")
        if status >= 400:
            raise SourceUnavailable(f"GNews returned HTTP {status}")
        try:
            data = response.json()
        except ValueError:
            raise SourceUnavailable("GNews returned a non-JSON response") from None
        return parse_gnews_response(data)
