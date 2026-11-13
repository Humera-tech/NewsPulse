"""URL normalization, date and relevance filters, and deduplication."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Protocol, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from rapidfuzz import fuzz

from newspulse.config import (
    ACADEMIC_URL_PATTERNS,
    FINANCE_SPAM_PATTERNS,
    MEDIA_URL_PATTERNS,
    TOPIC_SHORT_FORMS,
)
from newspulse.models import parse_iso, utc_now

TITLE_SIMILARITY_THRESHOLD = 90
SAME_HEADLINE_THRESHOLD = 60  # page title vs feed title, to reject unrelated page titles
RELEVANCE_WINDOW_CHARS = 500  # topic must appear in the title or this much of the text

_FINANCE_SPAM_RE = re.compile("|".join(f"(?:{p})" for p in FINANCE_SPAM_PATTERNS), re.IGNORECASE)
_SUFFIX_SEPARATORS = (" | ", " - ")

TRACKING_PARAMS = frozenset(
    {
        "fbclid", "gclid", "dclid", "msclkid", "yclid", "igshid", "mc_cid", "mc_eid",
        "ocid", "cmpid", "ref", "ref_src", "referrer", "smid", "smtyp",
        "ito", "taid", "_ga", "_gl", "guccounter", "guce_referrer", "guce_referrer_sig",
        "spm", "s_cid", "at_medium", "at_campaign",
    }
)

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into is
    it its itself just me more most my no nor not now of off on once only or other our
    ours out over own same she should so some such than that the their theirs them then
    there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours news latest
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")


class Dated(Protocol):
    """Anything with an ISO published_at attribute."""

    published_at: str | None


class Dedupable(Protocol):
    """Anything with a title, url and optional full_text."""

    title: str
    url: str
    full_text: str | None


T = TypeVar("T", bound=Dated)
D = TypeVar("D", bound=Dedupable)


def _is_tracking_param(key: str) -> bool:
    """Return True for utm_* and other known tracking query parameters."""
    lowered = key.lower()
    return lowered.startswith("utm_") or lowered in TRACKING_PARAMS


def clean_url(url: str) -> str:
    """Remove tracking parameters and the fragment but keep the URL otherwise fetchable."""
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not _is_tracking_param(k)])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def normalize_url(url: str) -> str:
    """Return a canonical key for a URL (for duplicate detection).

    Lowercases the host, drops "www.", tracking params, fragments and trailing slashes.
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not _is_tracking_param(k)])
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def is_within_days(published_at: str | None, days: int, now: datetime | None = None) -> bool:
    """Return True if the date is within `days` of now, or if there is no usable date."""
    published = parse_iso(published_at)
    if published is None:
        return True
    cutoff = (now or utc_now()) - timedelta(days=days)
    return published >= cutoff


def filter_by_date(items: Iterable[T], days: int, now: datetime | None = None) -> list[T]:
    """Drop items older than `days`; keep undated items (they get flagged later)."""
    return [item for item in items if is_within_days(item.published_at, days, now)]


def _stem(word: str) -> str:
    """Very small plural stemmer: batteries->battery, boxes->box, vehicles->vehicle."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    """Lowercase, stemmed word tokens of a text, in order."""
    return [_stem(w) for w in _WORD_RE.findall(text.lower())]


def topic_keywords(topic: str) -> list[str]:
    """Return the stemmed, non-stopword words of a topic."""
    words = _WORD_RE.findall(topic.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


def topic_phrase(topic: str) -> list[str]:
    """Stemmed topic words from the first to the last keyword (outer stopwords trimmed)."""
    words = _WORD_RE.findall(topic.lower())
    positions = [i for i, w in enumerate(words) if w not in STOPWORDS]
    if not positions:
        return []
    return [_stem(w) for w in words[positions[0]: positions[-1] + 1]]


def topic_phrases(topic: str) -> list[list[str]]:
    """The topic phrase plus any configured short forms (e.g. EV/EVs), as token lists."""
    phrase = topic_phrase(topic)
    short_forms = {" ".join(_tokens(k)): v for k, v in TOPIC_SHORT_FORMS.items()}
    aliases = short_forms.get(" ".join(phrase), ())
    return [phrase] + [_tokens(alias) for alias in aliases]


def contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    """Return True if `phrase` occurs as a contiguous run inside `tokens`."""
    n = len(phrase)
    if n == 0:
        return False
    return any(list(tokens[i: i + n]) == list(phrase) for i in range(len(tokens) - n + 1))


def is_relevant(topic: str, title: str | None, text: str | None) -> bool:
    """Return True if the article is about the topic.

    Multi-word topics: the full phrase (or a short form such as "EV") must appear
    in the title or the first 500 characters of the text.
    Single-word topics: the word must appear in the title or first 1500 characters.
    """
    keywords = topic_keywords(topic)
    if not keywords:
        return True
    if len(keywords) >= 2:
        parts = (_tokens(title or ""), _tokens((text or "")[:RELEVANCE_WINDOW_CHARS]))
        return any(contains_phrase(part, phrase) for part in parts for phrase in topic_phrases(topic))
    haystack = f"{title or ''} {(text or '')[:RELEVANCE_WINDOW_CHARS]}"
    present = set(_tokens(haystack))
    return all(keyword in present for keyword in keywords)


def is_finance_spam(title: str | None) -> bool:
    """Return True for auto-generated stock-ticker posts (see FINANCE_SPAM_PATTERNS)."""
    return bool(title) and bool(_FINANCE_SPAM_RE.search(title or ""))


# ---------------------------------------------------------------- titles


def publisher_slug(name: str | None) -> str:
    """Comparable publisher key: 'The Guardian' and 'theguardian.com' -> 'theguardian'."""
    value = (name or "").strip().lower().removeprefix("www.")
    if "." in value and " " not in value:
        value = value.split(".")[0]
    return re.sub(r"[^a-z0-9]", "", value)


def clean_title(title: str, publisher: str | None) -> str:
    """Drop breadcrumbs after ' > ' and a trailing ' | X' / ' - X' when X is the publisher."""
    cleaned = " ".join(title.split())
    if " > " in cleaned:
        cleaned = cleaned.split(" > ", 1)[0].strip()
    target = publisher_slug(publisher)
    changed = bool(target)
    while changed:
        changed = False
        for separator in _SUFFIX_SEPARATORS:
            head, sep, tail = cleaned.rpartition(separator)
            if sep and head.strip() and publisher_slug(tail) == target:
                cleaned, changed = head.strip(), True
    return cleaned or " ".join(title.split())


def repair_gdelt_spacing(title: str) -> str:
    """Undo GDELT's extra spaces: '( LON : SYS )' -> '(LON:SYS)', '2 . 8 %' -> '2.8%'."""
    text = " ".join(title.split())
    text = re.sub(r"\(\s+", "(", text)  # no space after (
    text = re.sub(r"\s+([,.:;)%!?])", r"\1", text)  # no space before , . : ; ) % ! ?
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)  # 2. 8 -> 2.8
    previous = None
    while previous != text:  # Ph. D. -> Ph.D. ; U. S. -> U.S.
        previous = text
        text = re.sub(r"\b([A-Z]|Ph)\.\s+(?=[A-Z]\.)", r"\1.", text)
    text = re.sub(r"\(([A-Z]{2,12}):\s+([A-Z0-9.\-]{1,10})\)", r"(\1:\2)", text)  # (LON: SYS)
    return text


def same_headline(a: str, b: str, threshold: int = SAME_HEADLINE_THRESHOLD) -> bool:
    """Return True if two titles describe the same headline (word-set similarity)."""
    return fuzz.token_set_ratio(a.lower(), b.lower()) >= threshold


def is_media_url(url: str, patterns: Sequence[str] = MEDIA_URL_PATTERNS) -> bool:
    """Return True for video, podcast, gallery and live pages (whole path segments only)."""
    segments = {s for s in urlsplit(url).path.lower().split("/") if s}
    return any(pattern.strip("/").lower() in segments for pattern in patterns)


def is_academic_url(url: str, patterns: Sequence[str] = ACADEMIC_URL_PATTERNS) -> bool:
    """Return True if the URL points at an academic paper site rather than news."""
    key = normalize_url(url).split("://", 1)[-1]  # host + path, no "www."
    host = key.split("/", 1)[0]
    for pattern in patterns:
        if "/" in pattern:
            if key == pattern or key.startswith(pattern + "/"):
                return True
        elif host == pattern or host.endswith("." + pattern):
            return True
    return False


def titles_similar(a: str, b: str, threshold: int = TITLE_SIMILARITY_THRESHOLD) -> bool:
    """Return True when two headlines are near-identical (token_sort_ratio > threshold)."""
    return fuzz.token_sort_ratio(a.lower(), b.lower()) > threshold


def _prefer(current: D, other: D) -> D:
    """Keep the version that has full text; otherwise keep the one seen first."""
    if not current.full_text and other.full_text:
        return other
    return current


def dedupe(items: Sequence[D]) -> list[D]:
    """Remove duplicates by normalized URL, then by near-identical title."""
    by_url: list[D] = []
    index: dict[str, int] = {}
    for item in items:
        key = normalize_url(item.url)
        if key in index:
            by_url[index[key]] = _prefer(by_url[index[key]], item)
        else:
            index[key] = len(by_url)
            by_url.append(item)

    unique: list[D] = []
    for item in by_url:
        for i, kept in enumerate(unique):
            if titles_similar(kept.title, item.title):
                unique[i] = _prefer(kept, item)
                break
        else:
            unique.append(item)
    return unique
