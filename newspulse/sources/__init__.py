"""News sources: GDELT, The Guardian, GNews and Google News RSS."""

from __future__ import annotations

import requests

from newspulse.config import Settings
from newspulse.fetcher import build_session
from newspulse.sources.base import Candidate, NewsSource, SourceUnavailable, safe_search
from newspulse.sources.gdelt import GdeltSource
from newspulse.sources.gnews import GNewsSource
from newspulse.sources.google_news import GoogleNewsSource
from newspulse.sources.guardian import GuardianSource

__all__ = [
    "Candidate",
    "GdeltSource",
    "GNewsSource",
    "GoogleNewsSource",
    "GuardianSource",
    "NewsSource",
    "SourceUnavailable",
    "build_sources",
    "safe_search",
]


def build_sources(session: requests.Session, settings: Settings) -> list[NewsSource]:
    """Create one instance of every available news source.

    GDELT (strict rate limit) and GNews (small daily quota, where every retry
    costs a request) get their own sessions without automatic retries.
    """
    return [
        GdeltSource(build_session(settings, retries=False), settings),
        GuardianSource(session, settings),
        GNewsSource(build_session(settings, retries=False), settings),
        GoogleNewsSource(session, settings),
    ]
