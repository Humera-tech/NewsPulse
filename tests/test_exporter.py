"""Tests for the digest/JSON output and per-source counts (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from newspulse.exporter import build_digest, export_results
from newspulse.models import Article, RunResult
from newspulse.pipeline import count_by_source

QUERIED = ["gdelt", "guardian", "gnews", "google_news"]


def article(via: str, method: str | None, text_type: str = "full_text", n: int = 1) -> Article:
    """Build a small article for digest tests."""
    return Article(
        title=f"Story {via} {n}", source="Example", url=f"https://example.com/{via}/{n}",
        published_at="2026-11-12T08:00:00Z", retrieved_at="2026-11-12T09:00:00Z",
        discovered_via=via,  # type: ignore[arg-type]
        excerpt="Some text." if text_type != "none" else None, text_type=text_type,  # type: ignore[arg-type]
        summary="A summary." if method else None, summary_method=method,  # type: ignore[arg-type]
    )


def run_result(articles: list[Article], summarizer: str = "gemini") -> RunResult:
    """Wrap articles in a RunResult the way the pipeline does."""
    return RunResult(
        topic="electric vehicles", generated_at="2026-11-12T09:30:00Z", days=7, requested_limit=5,
        article_count=len(articles), summarizer_requested=summarizer, sources_queried=QUERIED,
        articles_by_source=count_by_source(articles, QUERIED), articles=articles,
    )


def test_count_by_source_lists_only_contributors_in_source_order() -> None:
    """Sources with zero articles are left out; order follows the queried sources."""
    items = [article("google_news", "gemini", n=1), article("guardian", "gemini"),
             article("google_news", "gemini", n=2)]
    assert count_by_source(items, QUERIED) == {"guardian": 1, "google_news": 2}
    assert count_by_source([], QUERIED) == {}


def test_digest_separates_queried_sources_from_contributing_ones() -> None:
    """'Sources used' is replaced by 'Sources queried' and 'Articles from'."""
    digest = build_digest(run_result([article("google_news", "gemini", n=1),
                                      article("google_news", "gemini", n=2),
                                      article("gdelt", "gemini")]))
    assert "Sources used" not in digest
    assert "**Sources queried:** gdelt, guardian, gnews, google_news" in digest
    assert "**Articles from:** gdelt (1), google_news (2)" in digest


def test_digest_with_no_articles_says_none() -> None:
    """With no articles, 'Articles from' says none."""
    assert "**Articles from:** none" in build_digest(run_result([]))


def test_summary_labels() -> None:
    """Gemini, extractive fallback and deliberate extractive are labelled differently."""
    digest = build_digest(run_result([article("guardian", "gemini"),
                                      article("gnews", "extractive")], summarizer="gemini"))
    assert "*Summary: Gemini*" in digest
    assert "*Summary: extractive (Gemini unavailable)*" in digest

    chosen = build_digest(run_result([article("gnews", "extractive")], summarizer="extractive"))
    assert "*Summary: extractive*" in chosen
    assert "Gemini unavailable" not in chosen


def test_summary_label_comes_before_existing_warnings() -> None:
    """Excerpt-only and missing-field warnings are still shown after the label."""
    digest = build_digest(run_result([article("google_news", "extractive", text_type="feed_excerpt")]))
    label = digest.index("*Summary: extractive (Gemini unavailable)*")
    warning = digest.index("short feed excerpt")
    assert label < warning
    no_text = build_digest(run_result([article("gdelt", None, text_type="none")]))
    assert "*Summary:" not in no_text
    assert "no article text could be retrieved; missing fields: excerpt" in no_text.lower()


def test_json_has_articles_by_source(tmp_path: Path) -> None:
    """articles.json carries the same per-source counts as the digest."""
    result = run_result([article("guardian", "gemini"), article("google_news", "extractive")])
    paths = export_results(result, tmp_path, "json")
    data = json.loads(next(p for p in paths if p.name == "articles.json").read_text(encoding="utf-8"))
    assert data["articles_by_source"] == {"guardian": 1, "google_news": 1}
    assert data["sources_queried"] == QUERIED
    assert data["summarizer_requested"] == "gemini"
