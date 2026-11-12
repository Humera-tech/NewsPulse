"""Common interface for news sources and a wrapper that never lets one crash the run."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from newspulse.config import Settings

logger = logging.getLogger(__name__)


class SourceUnavailable(Exception):
    """Raised when a source cannot be used (missing key, bad response, ...)."""


@dataclass(kw_only=True)
class Candidate:
    """A possible article found by a source, before its page is fetched."""

    title: str
    url: str
    source: str | None
    published_at: str | None
    snippet: str | None
    full_text: str | None = None
    discovered_via: str


class NewsSource(ABC):
    """Abstract news source. Subclasses query one API or feed with plain HTTP."""

    name: str = "base"

    def __init__(self, session: requests.Session, settings: Settings) -> None:
        """Store the shared HTTP session and settings."""
        self.session = session
        self.settings = settings

    @abstractmethod
    def search(self, topic: str, days: int) -> list[Candidate]:
        """Return candidate articles about `topic` from the last `days` days."""

    def resolve_url(self, candidate: Candidate) -> str:
        """Return the real article URL for a candidate (default: unchanged)."""
        return candidate.url


def describe_error(exc: BaseException) -> str:
    """Turn an exception into a short, readable message for the user."""
    # urllib3 retries wrap read timeouts in a ConnectionError, so also check the message.
    if isinstance(exc, requests.Timeout) or "timed out" in str(exc).lower():
        return "request timed out"
    if isinstance(exc, requests.ConnectionError):
        return "could not connect (network error or too many retries)"
    if isinstance(exc, requests.RequestException):
        return f"HTTP request failed ({type(exc).__name__})"
    return str(exc).strip() or type(exc).__name__


def safe_search(source: NewsSource, topic: str, days: int) -> tuple[list[Candidate], str | None]:
    """Run source.search() and return (candidates, error_message) without raising."""
    try:
        candidates = source.search(topic, days)
    except Exception as exc:  # noqa: BLE001 - one failing source must never stop the run
        message = describe_error(exc)
        logger.debug("Source %s error detail: %r", source.name, exc)
        logger.warning("Source %s failed: %s", source.name, message)
        return [], message
    logger.info("Source %s returned %d candidates", source.name, len(candidates))
    return candidates, None
