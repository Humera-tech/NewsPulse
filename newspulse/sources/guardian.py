"""The Guardian Open Platform API source (free key in GUARDIAN_API_KEY)."""

from __future__ import annotations

import html
import re
from datetime import timedelta

from newspulse.models import parse_iso, to_iso_utc, utc_now
from newspulse.sources.base import Candidate, NewsSource, SourceUnavailable

GUARDIAN_URL = "https://content.guardianapis.com/search"
PUBLISHER = "The Guardian"
PAGE_SIZE = 20
_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(value: str | None) -> str | None:
    """Remove HTML tags and entities and collapse whitespace."""
    if not value:
        return None
    text = " ".join(html.unescape(_TAG_RE.sub(" ", value)).split())
    return text or None


def parse_guardian_response(data: dict) -> list[Candidate]:
    """Turn a Guardian /search JSON payload into candidates (with full text)."""
    response = data.get("response") if isinstance(data, dict) else None
    if not isinstance(response, dict):
        raise SourceUnavailable("Guardian returned an unexpected JSON structure")
    if response.get("status") not in (None, "ok"):
        raise SourceUnavailable(f"Guardian API error: {response.get('message', 'unknown error')}")

    candidates: list[Candidate] = []
    for item in response.get("results") or []:
        url = (item.get("webUrl") or "").strip()
        title = " ".join((item.get("webTitle") or "").split())
        if not url or not title:
            continue
        fields = item.get("fields") or {}
        published = parse_iso(item.get("webPublicationDate"))
        candidates.append(
            Candidate(
                title=title,
                url=url,
                source=PUBLISHER,
                published_at=to_iso_utc(published) if published else None,
                snippet=strip_html(fields.get("trailText")),
                full_text=(fields.get("bodyText") or "").strip() or None,
                discovered_via="guardian",
            )
        )
    return candidates


class GuardianSource(NewsSource):
    """Searches The Guardian's content API, which also returns full article text."""

    name = "guardian"

    def search(self, topic: str, days: int) -> list[Candidate]:
        """Query the Guardian API for the newest articles about the topic."""
        api_key = self.settings.guardian_api_key
        if not api_key:
            raise SourceUnavailable("GUARDIAN_API_KEY not set")
        params = {
            "q": topic,
            "type": "article",  # excludes live blogs, galleries, video and audio
            "from-date": (utc_now() - timedelta(days=days)).strftime("%Y-%m-%d"),
            "order-by": "newest",
            "page-size": str(PAGE_SIZE),
            "show-fields": "bodyText,trailText",
            "api-key": api_key,
        }
        response = self.session.get(GUARDIAN_URL, params=params, timeout=self.settings.request_timeout)
        if response.status_code in (401, 403):
            raise SourceUnavailable(f"Guardian rejected the API key (HTTP {response.status_code})")
        if response.status_code >= 400:
            raise SourceUnavailable(f"Guardian returned HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise SourceUnavailable("Guardian returned a non-JSON response") from None
        return parse_guardian_response(data)
