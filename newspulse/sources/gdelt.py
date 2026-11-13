"""GDELT DOC 2.0 API source (no API key needed)."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import requests

from newspulse.models import to_iso_utc
from newspulse.sources.base import Candidate, NewsSource, SourceUnavailable

logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 75
GDELT_TIMEOUT = 30.0  # GDELT often takes 15-25 s to answer
RATE_LIMIT_WAIT = 6.0


def _body_for_log(text: str | None) -> str:
    """Whole response body on one line, for the DEBUG log."""
    return " ".join((text or "").split()) or "(empty body)"


def build_query(topic: str) -> str:
    """Build the GDELT query string, quoting multi-word topics as a phrase."""
    cleaned = " ".join(topic.replace('"', " ").split())
    phrase = f'"{cleaned}"' if " " in cleaned else cleaned
    return f"{phrase} sourcelang:english"


def parse_seendate(value: str | None) -> str | None:
    """Convert GDELT's 'YYYYMMDDTHHMMSSZ' seendate into ISO 8601 UTC."""
    if not value:
        return None
    try:
        dt = datetime.strptime(value.strip(), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return to_iso_utc(dt)


def parse_gdelt_response(body: str) -> list[Candidate]:
    """Parse a GDELT ArtList JSON body. Raises SourceUnavailable for non-JSON replies."""
    text = (body or "").strip()
    if not text:
        return []  # GDELT returns an empty body when nothing matches
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("GDELT non-JSON response body: %s", _body_for_log(text))
        raise SourceUnavailable("GDELT returned a non-JSON response") from None
    if not isinstance(data, dict):
        raise SourceUnavailable("GDELT returned an unexpected JSON structure")

    candidates: list[Candidate] = []
    for item in data.get("articles") or []:
        url = (item.get("url") or "").strip()
        title = " ".join((item.get("title") or "").split())
        if not url or not title:
            continue
        candidates.append(
            Candidate(
                title=title,
                url=url,
                source=(item.get("domain") or "").strip() or None,
                published_at=parse_seendate(item.get("seendate")),
                snippet=None,
                discovered_via="gdelt",
            )
        )
    return candidates


class GdeltSource(NewsSource):
    """Searches the GDELT DOC 2.0 article list."""

    name = "gdelt"

    def _get(self, params: dict[str, str]) -> requests.Response:
        """GET the API, retrying once after a pause if GDELT asks us to slow down.

        GDELT allows about one request every 5 seconds and is often slow to
        answer, so this uses a longer read timeout and its own gentle retry
        instead of the shared fast-retry policy.
        """
        # GDELT can stall during the TLS handshake too, so both connect and read get the long timeout.
        timeout = max(GDELT_TIMEOUT, self.settings.request_timeout)
        response = self._request_once(params, timeout, attempt=1)
        if response.status_code == 429:
            logger.info("GDELT rate limit hit; waiting %.0fs before one retry", RATE_LIMIT_WAIT)
            time.sleep(RATE_LIMIT_WAIT)
            response = self._request_once(params, timeout, attempt=2)
        return response

    def _request_once(self, params: dict[str, str], timeout: float, attempt: int) -> requests.Response:
        """Send exactly one GET to GDELT, logging the final URL, status and any 429 body."""
        started = time.monotonic()
        response = self.session.get(GDELT_URL, params=params, timeout=timeout)
        final_url = getattr(response, "url", None) or GDELT_URL
        logger.info("GDELT request %d: GET %s -> HTTP %s in %.1fs", attempt, final_url,
                    response.status_code, time.monotonic() - started)
        if response.status_code == 429:
            logger.debug("GDELT 429 response body: %s", _body_for_log(response.text))
        return response

    def search(self, topic: str, days: int) -> list[Candidate]:
        """Query GDELT for English articles about the topic within the last `days` days."""
        params = {
            "query": build_query(topic),
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(MAX_RECORDS),
            "timespan": f"{days}d",
            "sort": "DateDesc",
        }
        response = self._get(params)
        if response.status_code == 429:
            raise SourceUnavailable("GDELT rate limit reached (HTTP 429)")
        if response.status_code >= 400:
            raise SourceUnavailable(f"GDELT returned HTTP {response.status_code}")
        return parse_gdelt_response(response.text)
