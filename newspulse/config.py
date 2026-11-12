"""Settings loaded from environment variables and an optional .env file."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

USER_AGENT = "NewsPulse/1.0 (educational news digest project)"
PER_DOMAIN_DELAY = 1.0
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_REQUEST_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 2

# URLs that point at academic papers rather than news. An entry with a "/" matches
# host + path prefix; an entry without one matches the domain and its subdomains.
ACADEMIC_URL_PATTERNS: tuple[str, ...] = (
    "nature.com/articles",
    "arxiv.org",
    "sciencedirect.com",
    "springer.com",
    "mdpi.com",
    "ieeexplore.ieee.org",
    "researchgate.net",
    "biorxiv.org",
    "medrxiv.org",
    "ssrn.com",
    "pubmed.ncbi.nlm.nih.gov",
    "tandfonline.com",
    "onlinelibrary.wiley.com",
)

# Common short forms accepted in place of a multi-word topic phrase (case-insensitive).
TOPIC_SHORT_FORMS: dict[str, tuple[str, ...]] = {
    "electric vehicles": ("EV", "EVs"),
    "artificial intelligence": ("AI",),
    "united states": ("USA", "U.S."),  # not "US": it would match the word "us"
    "united kingdom": ("UK", "U.K."),
    "european union": ("EU", "E.U."),
}


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for one NewsPulse run."""

    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    guardian_api_key: str | None = None
    gnews_api_key: str | None = None
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    user_agent: str = USER_AGENT
    per_domain_delay: float = PER_DOMAIN_DELAY


def _env_str(name: str) -> str | None:
    """Return a stripped environment variable, or None if unset or blank."""
    value = os.getenv(name, "").strip()
    return value or None


def _env_number(name: str, default: float, cast: type) -> float:
    """Read a numeric environment variable, falling back to `default` if invalid."""
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if value < 0:
        logger.warning("Ignoring negative %s=%r; using %s", name, raw, default)
        return default
    return value


def load_settings() -> Settings:
    """Load .env (if present) and build a Settings object from the environment."""
    load_dotenv()
    return Settings(
        gemini_api_key=_env_str("GEMINI_API_KEY"),
        gemini_model=_env_str("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        guardian_api_key=_env_str("GUARDIAN_API_KEY"),
        gnews_api_key=_env_str("GNEWS_API_KEY"),
        request_timeout=float(_env_number("REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT, float)),
        max_retries=int(_env_number("MAX_RETRIES", DEFAULT_MAX_RETRIES, int)),
    )
