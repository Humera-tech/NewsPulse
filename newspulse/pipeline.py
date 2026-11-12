"""The NewsPulse workflow: discover -> clean -> retrieve -> extract -> summarize."""

from __future__ import annotations

import logging
import re
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from newspulse.config import Settings, load_settings
from newspulse.extractor import ExtractedText, choose_text, extract_article
from newspulse.fetcher import Fetcher, build_session
from newspulse.filters import (
    clean_url,
    dedupe,
    filter_by_date,
    is_academic_url,
    is_relevant,
    is_within_days,
    normalize_url,
)
from newspulse.models import Article, RunResult, SourceError, fill_missing_fields, parse_iso, utc_now_iso
from newspulse.sources import Candidate, NewsSource, build_sources, safe_search
from newspulse.sources.google_news import is_google_news_link
from newspulse.summarizer import Summarizer, build_summarizer

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]
_OLDEST = datetime.min.replace(tzinfo=timezone.utc)

SOURCE_ORDER: tuple[str, ...] = ("guardian", "gdelt", "gnews", "google_news")
COVERAGE_ATTEMPTS_PER_SOURCE = 3


@dataclass
class CollectStats:
    """Counters explaining why candidates were not used."""

    checked: int = 0
    not_relevant: int = 0
    out_of_range: int = 0
    duplicates: int = 0
    academic: int = 0
    no_text: int = 0
    attempt_cap_hit: bool = False
    backups: list[Article] = field(default_factory=list)


@dataclass
class CollectContext:
    """Everything needed to turn candidates into articles during one run."""

    topic: str
    days: int
    fetcher: Fetcher
    sources: dict[str, NewsSource]
    stats: CollectStats = field(default_factory=CollectStats)
    seen: set[str] = field(default_factory=set)


def _noop(_: str) -> None:
    """Default progress callback that does nothing."""


def discover(
    sources: Sequence[NewsSource], topic: str, days: int, progress: ProgressCallback = _noop
) -> tuple[list[Candidate], list[SourceError]]:
    """Query every source safely and return all candidates plus source errors."""
    candidates: list[Candidate] = []
    errors: list[SourceError] = []
    for source in sources:
        progress(f"Searching {source.name}...")
        found, error = safe_search(source, topic, days)
        candidates.extend(found)
        if error:
            errors.append(SourceError(source=source.name, error=error))
    return candidates, errors


def publisher_key(candidate: Candidate) -> str:
    """Group key for a publisher: 'The Guardian' and 'theguardian.com' both -> 'theguardian'."""
    name = (candidate.source or normalize_url(candidate.url).split("/")[2]).lower()
    name = name.removeprefix("www.")
    if "." in name and " " not in name:
        name = name.split(".")[0]
    return re.sub(r"[^a-z0-9]", "", name) or "unknown"


def sort_newest_first(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Sort by publication date, newest first; undated candidates go last."""
    return sorted(candidates, key=lambda c: parse_iso(c.published_at) or _OLDEST, reverse=True)


def round_robin(groups: Sequence[Sequence[Candidate]]) -> list[Candidate]:
    """Take the first item of each group, then the second of each, and so on."""
    result: list[Candidate] = []
    longest = max((len(group) for group in groups), default=0)
    for depth in range(longest):
        result.extend(group[depth] for group in groups if depth < len(group))
    return result


def interleave_by_publisher(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Round-robin across publishers, keeping each publisher's own order."""
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(publisher_key(candidate), []).append(candidate)
    return round_robin(list(groups.values()))


def _source_rank(name: str) -> int:
    """Position of a source in SOURCE_ORDER (unknown sources go last)."""
    return SOURCE_ORDER.index(name) if name in SOURCE_ORDER else len(SOURCE_ORDER)


def interleave_by_source(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Round-robin across sources (guardian, gdelt, gnews, google_news), then by publisher."""
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.discovered_via, []).append(candidate)
    ordered = sorted(groups, key=_source_rank)
    return round_robin([interleave_by_publisher(groups[name]) for name in ordered])


def prepare_candidates(candidates: Sequence[Candidate], days: int) -> list[Candidate]:
    """Clean URLs, drop academic papers, dedupe, drop old items, then mix sources/publishers."""
    for candidate in candidates:
        candidate.url = clean_url(candidate.url)
    news = [c for c in candidates if not is_academic_url(c.url)]
    unique = dedupe(news)
    recent = filter_by_date(unique, days)
    logger.info("%d candidates -> %d news -> %d after dedupe -> %d within %d days",
                len(candidates), len(news), len(unique), len(recent), days)
    return interleave_by_source(sort_newest_first(recent))


def _get_text(candidate: Candidate, url: str, fetcher: Fetcher) -> ExtractedText:
    """Return the article text: Guardian full text directly, otherwise fetch + extract."""
    if candidate.full_text:
        return choose_text(candidate.full_text, candidate.snippet)
    if is_google_news_link(url):
        # Link could not be decoded; Google's page is not the article, so use the snippet.
        return choose_text(None, candidate.snippet)
    html = fetcher.fetch_html(url)
    return extract_article(html, candidate.snippet, need_date=candidate.published_at is None)


def build_article(candidate: Candidate, ctx: CollectContext) -> Article | None:
    """Turn one candidate into an Article, or return None (and count why) if unusable."""
    stats = ctx.stats
    source = ctx.sources.get(candidate.discovered_via)
    url = clean_url(source.resolve_url(candidate) if source else candidate.url)
    if is_academic_url(url):  # decoded Google links are only known at this point
        stats.academic += 1
        return None
    key = normalize_url(url)
    if key in ctx.seen:
        stats.duplicates += 1
        return None

    extracted = _get_text(candidate, url, ctx.fetcher)
    retrieved_at = utc_now_iso()
    published_at = candidate.published_at or extracted.published_at
    if not is_within_days(published_at, ctx.days):
        stats.out_of_range += 1
        return None
    if not is_relevant(ctx.topic, candidate.title, extracted.text):
        stats.not_relevant += 1
        return None

    ctx.seen.update({key, normalize_url(candidate.url)})
    return Article(
        title=candidate.title,
        source=candidate.source,
        url=url,
        published_at=published_at,
        retrieved_at=retrieved_at,
        discovered_via=candidate.discovered_via,  # type: ignore[arg-type]
        excerpt=extracted.text,
        text_type=extracted.text_type,  # type: ignore[arg-type]
    )


def try_candidate(candidate: Candidate, ctx: CollectContext) -> Article | None:
    """Process one candidate; return an Article with text, or None. Never raises."""
    ctx.stats.checked += 1
    try:
        article = build_article(candidate, ctx)
    except Exception as exc:  # noqa: BLE001 - one bad page must never crash the run
        logger.warning("Skipped %s: unexpected error (%s)", candidate.url, exc)
        return None
    if article is not None and article.text_type == "none":
        ctx.stats.no_text += 1
        ctx.stats.backups.append(article)  # real but text-less; used only to fill a shortfall
        return None
    return article


def ensure_source_coverage(
    articles: list[Article], pending: deque[Candidate], limit: int, ctx: CollectContext
) -> None:
    """Make sure every source with a relevant article is represented in the results.

    For each source that has unchecked candidates but no article yet, try a few of
    its candidates; if one works, it replaces the last article from the source with
    the most articles (only when that source has more than one).
    """
    for name in sorted({c.discovered_via for c in pending}, key=_source_rank):
        if any(a.discovered_via == name for a in articles):
            continue
        tries = [c for c in pending if c.discovered_via == name][:COVERAGE_ATTEMPTS_PER_SOURCE]
        for candidate in tries:
            pending.remove(candidate)
            article = try_candidate(candidate, ctx)
            if article is None:
                continue
            if len(articles) >= limit:
                donor, count = Counter(a.discovered_via for a in articles).most_common(1)[0]
                if count < 2:
                    break  # replacing would remove another source's only article
                last = max(i for i, a in enumerate(articles) if a.discovered_via == donor)
                articles.pop(last)
            articles.append(article)
            logger.info("Added an article from %s so every source is represented", name)
            break


def collect_articles(
    candidates: Sequence[Candidate],
    topic: str,
    limit: int,
    days: int,
    fetcher: Fetcher,
    sources: dict[str, NewsSource],
    progress: ProgressCallback = _noop,
) -> tuple[list[Article], CollectStats]:
    """Process candidates in order until `limit` articles with text are collected."""
    ctx = CollectContext(topic=topic, days=days, fetcher=fetcher, sources=sources)
    articles: list[Article] = []
    pending = deque(candidates)
    max_attempts = max(limit * 4, 20)

    while pending and len(articles) < limit:
        if ctx.stats.checked >= max_attempts:
            ctx.stats.attempt_cap_hit = True
            break
        progress(f"Retrieving articles ({len(articles)}/{limit} collected, "
                 f"{ctx.stats.checked} checked)...")
        article = try_candidate(pending.popleft(), ctx)
        if article is not None:
            articles.append(article)

    progress("Checking that every source is represented...")
    ensure_source_coverage(articles, pending, limit, ctx)
    articles.extend(ctx.stats.backups[: max(0, limit - len(articles))])
    return articles, ctx.stats


def summarize_articles(
    articles: Sequence[Article], summarizer: Summarizer, progress: ProgressCallback = _noop
) -> None:
    """Add a summary to each article in place; flag 'summary' when there is no text."""
    for index, article in enumerate(articles, start=1):
        progress(f"Summarizing article {index}/{len(articles)}...")
        summary, method = summarizer.summarize(article.title or "", article.excerpt)
        article.summary = summary
        article.summary_method = method  # type: ignore[assignment]
        if summary is None:
            article.missing_fields.append("summary")
        fill_missing_fields(article)


def build_shortfall_note(
    count: int,
    limit: int,
    days: int,
    total_candidates: int,
    stats: CollectStats,
    errors: Sequence[SourceError],
) -> str | None:
    """Explain why fewer articles than requested were returned (None if no shortfall)."""
    if count >= limit:
        return None
    day_word = "day" if days == 1 else "days"
    if count == 0:
        lead = f"No relevant articles from the last {days} {day_word} were found"
    else:
        noun = "article was" if count == 1 else "articles were"
        lead = f"Only {count} relevant {noun} from the last {days} {day_word} found"
    parts = [f"{lead} ({limit} requested)"]

    if total_candidates == 0:
        parts.append("the sources returned no usable candidates")
    else:
        reasons = [
            f"{stats.not_relevant} not relevant" if stats.not_relevant else "",
            f"{stats.out_of_range} outside the date range" if stats.out_of_range else "",
            f"{stats.duplicates} duplicates" if stats.duplicates else "",
            f"{stats.academic} academic papers" if stats.academic else "",
            f"{stats.no_text} without retrievable text" if stats.no_text else "",
        ]
        detail = ", ".join(r for r in reasons if r)
        parts.append(f"{stats.checked} of {total_candidates} candidates checked"
                     + (f" ({detail})" if detail else ""))
    if stats.attempt_cap_hit:
        parts.append("stopped early to keep the run time reasonable")
    if errors:
        failed = ", ".join(f"{e.source} ({e.error})" for e in errors)
        parts.append(f"{len(errors)} source{'s' if len(errors) != 1 else ''} failed: {failed}")
    return "; ".join(parts) + "."


def run_pipeline(
    topic: str,
    limit: int,
    days: int,
    summarizer_name: str,
    settings: Settings | None = None,
    progress: ProgressCallback = _noop,
) -> RunResult:
    """Run the full workflow for one topic and return the structured result."""
    settings = settings or load_settings()
    session = build_session(settings)
    try:
        sources = build_sources(session, settings)
        raw, errors = discover(sources, topic, days, progress)
        candidates = prepare_candidates(raw, days)

        fetcher = Fetcher(session, settings)
        by_name = {source.name: source for source in sources}
        articles, stats = collect_articles(candidates, topic, limit, days, fetcher, by_name, progress)

        summarize_articles(articles, build_summarizer(summarizer_name, settings), progress)
    finally:
        session.close()

    return RunResult(
        topic=topic,
        generated_at=utc_now_iso(),
        days=days,
        requested_limit=limit,
        article_count=len(articles),
        sources_queried=[source.name for source in sources],
        source_errors=errors,
        shortfall_note=build_shortfall_note(len(articles), limit, days, len(candidates), stats, errors),
        articles=articles,
    )
