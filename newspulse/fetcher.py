"""Polite HTTP fetching: one session, retries, per-domain delay and robots.txt."""

from __future__ import annotations

import logging
import time
from urllib import robotparser
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from newspulse.config import Settings

logger = logging.getLogger(__name__)

RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_RETRY_AFTER_SECONDS = 10.0
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


class _CappedRetry(Retry):
    """urllib3 Retry that honours Retry-After but never waits longer than a cap."""

    def get_retry_after(self, response) -> float | None:  # type: ignore[override]
        """Return the server's Retry-After value, capped at MAX_RETRY_AFTER_SECONDS."""
        value = super().get_retry_after(response)
        return None if value is None else min(value, MAX_RETRY_AFTER_SECONDS)


def build_session(settings: Settings, retries: bool = True) -> requests.Session:
    """Create a requests.Session with the NewsPulse user agent and retry policy.

    Pass retries=False for APIs with strict rate limits that handle retries themselves.
    """
    attempts = settings.max_retries if retries else 0
    retry = _CappedRetry(
        total=attempts,
        connect=attempts,
        read=attempts,
        status=attempts,
        backoff_factor=1.0,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        }
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _domain(url: str) -> str:
    """Return the lowercase network location of a URL."""
    return urlsplit(url).netloc.lower()


class Fetcher:
    """Fetches article pages politely. State is per instance, never global."""

    def __init__(self, session: requests.Session, settings: Settings) -> None:
        """Store the session/settings and set up per-domain caches."""
        self._session = session
        self._settings = settings
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._last_request: dict[str, float] = {}

    def _wait_for_domain(self, domain: str) -> None:
        """Sleep so requests to the same domain are PER_DOMAIN_DELAY seconds apart."""
        last = self._last_request.get(domain)
        if last is not None:
            remaining = self._settings.per_domain_delay - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request[domain] = time.monotonic()

    def _load_robots(self, url: str) -> robotparser.RobotFileParser | None:
        """Download and parse robots.txt for the URL's site (None = no restrictions known)."""
        parts = urlsplit(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        parser = robotparser.RobotFileParser(robots_url)
        self._wait_for_domain(parts.netloc.lower())
        try:
            response = self._session.get(robots_url, timeout=self._settings.request_timeout)
        except requests.RequestException as exc:
            logger.debug("Could not read %s (%s); assuming allowed", robots_url, exc)
            return None
        if response.status_code in (401, 403):
            parser.disallow_all = True
        elif response.status_code >= 400:
            return None
        else:
            parser.parse(response.text.splitlines())
        return parser

    def is_allowed(self, url: str) -> bool:
        """Return True if robots.txt allows NewsPulse to fetch this URL (cached per domain)."""
        domain = _domain(url)
        if domain not in self._robots:
            self._robots[domain] = self._load_robots(url)
        parser = self._robots[domain]
        return True if parser is None else parser.can_fetch(self._settings.user_agent, url)

    def fetch_html(self, url: str) -> str | None:
        """Fetch a page and return its HTML, or None with a logged reason on failure.

        Paywalls, logins and CAPTCHAs are never bypassed: a blocked page is
        simply reported and skipped.
        """
        if urlsplit(url).scheme not in ("http", "https"):
            logger.warning("Skipped %s: not an http(s) URL", url)
            return None
        if not self.is_allowed(url):
            logger.warning("Skipped %s: disallowed by robots.txt", url)
            return None

        self._wait_for_domain(_domain(url))
        try:
            response = self._session.get(url, timeout=self._settings.request_timeout)
        except requests.RequestException as exc:
            if isinstance(exc, requests.Timeout) or "timed out" in str(exc).lower():
                logger.warning("Skipped %s: timed out after %ss", url, self._settings.request_timeout)
                return None
            logger.warning("Skipped %s: connection failed (%s)", url, type(exc).__name__)
            return None

        return self._html_from_response(url, response)

    @staticmethod
    def _html_from_response(url: str, response: requests.Response) -> str | None:
        """Validate status and content type of a response and return its text."""
        status = response.status_code
        if status in (401, 402, 403):
            logger.warning("Skipped %s: blocked or requires login (%s)", url, status)
            return None
        if status == 404:
            logger.warning("Skipped %s: not found (404)", url)
            return None
        if status == 429:
            logger.warning("Skipped %s: rate limited (429)", url)
            return None
        if status >= 400:
            logger.warning("Skipped %s: HTTP error %s", url, status)
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not any(t in content_type for t in HTML_CONTENT_TYPES):
            logger.warning("Skipped %s: non-HTML content (%s)", url, content_type.split(";")[0])
            return None

        if response.encoding is None or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding
        return response.text
