"""Tests for URL normalization, date/relevance filters and deduplication."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from newspulse.filters import (
    clean_url,
    dedupe,
    filter_by_date,
    is_academic_url,
    is_relevant,
    normalize_url,
)
from newspulse.pipeline import (
    collect_articles,
    interleave_by_publisher,
    interleave_by_source,
    prepare_candidates,
    publisher_key,
)
from newspulse.sources.base import Candidate
from newspulse.sources.gdelt import parse_gdelt_response
from newspulse.sources.gnews import parse_gnews_response
from tests.conftest import read_fixture

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)


def make(title: str, url: str, full_text: str | None = None, published_at: str | None = None,
         source: str | None = "example.com") -> Candidate:
    """Build a candidate for tests."""
    return Candidate(title=title, url=url, source=source, published_at=published_at,
                     snippet=None, full_text=full_text, discovered_via="gdelt")


# ------------------------------------------------------- normalize_url


def test_normalize_url_removes_noise() -> None:
    """Host lowercased, www dropped, tracking params, fragment and trailing slash removed."""
    url = "HTTPS://WWW.Example.com/News/Story/?utm_source=x&id=7&fbclid=abc#comments"
    assert normalize_url(url) == "https://example.com/News/Story?id=7"


def test_normalize_url_equal_for_variants() -> None:
    """Variants of the same article share one key."""
    a = normalize_url("https://www.example.com/a/b/")
    b = normalize_url("https://example.com/a/b?utm_campaign=c")
    assert a == b


def test_clean_url_keeps_www() -> None:
    """clean_url drops tracking but keeps the URL fetchable as-is."""
    assert clean_url("https://www.example.com/a?utm_medium=x#top") == "https://www.example.com/a"


# ------------------------------------------------------------- dedupe


def test_dedupe_same_url_with_utm_params() -> None:
    """The same URL with tracking params counts once, keeping the full-text version."""
    items = [
        make("Story one", "https://www.example.com/story?utm_source=a"),
        make("Story one (copy)", "https://example.com/story/", full_text="full body"),
        make("Something else entirely", "https://example.com/other"),
    ]
    result = dedupe(items)
    assert len(result) == 2
    assert result[0].full_text == "full body"


def test_dedupe_near_identical_titles() -> None:
    """Near-identical headlines on different sites are merged; the full-text one wins."""
    items = [
        make("Battery plant for electric vehicles opens in Ohio", "https://a.example.com/1"),
        make("Battery plant for electric vehicles opens in Ohio.", "https://b.example.com/2",
             full_text="text"),
        make("Battery plant closes in Texas", "https://c.example.com/3"),
    ]
    result = dedupe(items)
    assert [c.url for c in result] == ["https://b.example.com/2", "https://c.example.com/3"]


def test_dedupe_keeps_order_of_first_appearance() -> None:
    """Deduplication does not reorder unique items."""
    items = [make("Alpha headline here", "https://x.example/1"),
             make("Totally different beta", "https://x.example/2")]
    assert dedupe(items) == items


# --------------------------------------------------------- date filter


def test_filter_by_date_drops_old_keeps_undated() -> None:
    """Old items are dropped; undated items are kept (and flagged later)."""
    candidates = parse_gdelt_response(read_fixture("gdelt.json"))
    candidates.append(make("Undated story", "https://example.com/u"))
    kept = filter_by_date(candidates, days=7, now=NOW)
    titles = [c.title for c in kept]
    assert "Electric vehicle sales figures from last quarter" not in titles
    assert "Undated story" in titles
    assert len(kept) == 3


def test_filter_by_date_boundary() -> None:
    """An item exactly `days` old is still included."""
    item = make("Edge", "https://e.example", published_at="2026-11-05T12:00:00Z")
    assert filter_by_date([item], days=7, now=NOW) == [item]
    assert filter_by_date([item], days=6, now=NOW) == []


# ---------------------------------------------------- relevance filter


def test_relevance_all_words_required() -> None:
    """Every non-stopword topic word must appear."""
    assert is_relevant("electric vehicles", "New electric vehicles unveiled", None)
    assert not is_relevant("electric vehicles", "New electric scooters unveiled", "Scooters only.")


def test_relevance_plural_and_case() -> None:
    """Simple plural handling and case-insensitivity."""
    assert is_relevant("Electric Vehicles", "An electric vehicle maker", None)
    assert is_relevant("battery", "Batteries are getting cheaper", None)
    assert is_relevant("batteries", "A battery breakthrough", None)


def test_relevance_phrase_in_start_of_text() -> None:
    """A multi-word phrase may appear in the first 500 characters of the text."""
    text = "The city announced a plan to buy electric vehicles for its fleet."
    assert is_relevant("electric vehicles", "City plan announced", text)
    assert not is_relevant("electric vehicles", "Title", "x " * 300 + "electric vehicles")


def test_relevance_scattered_words_are_not_enough() -> None:
    """Individual topic words in the text do not count for a multi-word topic."""
    text = "The city bought electric buses. Other vehicles will be replaced later."
    assert not is_relevant("electric vehicles", "City transport plan", text)
    assert not is_relevant("electric vehicles", "New Vehicle Prices Hit Record High",
                           "Prices for new vehicles rose. Electric models were cheaper.")


def test_relevance_short_forms() -> None:
    """Common short forms such as EV/EVs count for 'electric vehicles'."""
    assert is_relevant("electric vehicles", "Global EV sales rise 8% in October", None)
    assert is_relevant("Electric Vehicles", "Why EVs are getting cheaper", None)
    assert is_relevant("electric vehicles", "Report", "Demand for EVs grew in Europe this year.")
    assert not is_relevant("electric vehicles", "Every vehicle counts", None)


def test_relevance_trims_outer_stopwords_from_phrase() -> None:
    """'the future of electric vehicles' needs 'future of electric vehicles' in order."""
    assert is_relevant("the future of electric vehicles", "On the future of electric vehicles", None)
    assert not is_relevant("the future of electric vehicles", "Electric vehicles: the future", None)


def test_single_word_topic_uses_1500_char_window() -> None:
    """Single-word topics still search the title and the first 1500 characters."""
    assert is_relevant("cybersecurity", "Bank report", "x " * 400 + "cybersecurity spending rose")
    assert not is_relevant("cybersecurity", "Bank report", "x " * 800 + "cybersecurity")


# ------------------------------------------------------- academic URLs


def test_academic_urls_are_detected() -> None:
    """Paper repositories and journals are recognised, including subdomains."""
    for url in (
        "https://www.nature.com/articles/s41586-026-01234-5",
        "https://arxiv.org/abs/2611.01234",
        "https://www.sciencedirect.com/science/article/pii/S0000",
        "https://link.springer.com/article/10.1007/x",
        "https://www.mdpi.com/1996-1073/19/4/123",
        "https://ieeexplore.ieee.org/document/1234567",
    ):
        assert is_academic_url(url), url


def test_news_urls_are_not_academic() -> None:
    """News pages, including nature.com news, are kept."""
    assert not is_academic_url("https://www.nature.com/news/some-story")
    assert not is_academic_url("https://www.example.com/articles/ev-sales")
    assert not is_academic_url("https://notarxiv.org.example.com/story")


def test_prepare_candidates_drops_academic_papers() -> None:
    """Academic paper URLs never reach the fetch stage."""
    items = [make("EV paper", "https://arxiv.org/abs/1", published_at="2026-11-11T00:00:00Z"),
             make("EV news", "https://example.com/ev", published_at="2026-11-11T00:00:00Z")]
    assert [c.title for c in prepare_candidates(items, days=30)] == ["EV news"]


def test_relevance_whole_words_only() -> None:
    """'AI' must not match inside other words."""
    assert not is_relevant("AI", "Airline profits rise", "Said the chairman.")
    assert is_relevant("AI", "New AI rules agreed", None)


# ------------------------------------------------------ publisher mix


def test_publisher_key_matches_domain_and_name() -> None:
    """'The Guardian' and 'theguardian.com' group together."""
    a = make("a", "https://www.theguardian.com/x", source="The Guardian")
    b = make("b", "https://www.theguardian.com/y", source="theguardian.com")
    assert publisher_key(a) == publisher_key(b) == "theguardian"


def test_interleave_by_publisher() -> None:
    """Results alternate between publishers instead of clustering."""
    items = [make("1", "https://a.example/1", source="A"), make("2", "https://a.example/2", source="A"),
             make("3", "https://b.example/3", source="B")]
    assert [c.title for c in interleave_by_publisher(items)] == ["1", "3", "2"]


def via(title: str, discovered_via: str, source: str, full_text: str | None = None) -> Candidate:
    """Build a candidate from a given source."""
    return Candidate(title=title, url=f"https://{source.lower()}.example/{title.replace(' ', '-')}",
                     source=source, published_at=None, snippet=None, full_text=full_text,
                     discovered_via=discovered_via)


def test_interleave_by_source_then_publisher() -> None:
    """Sources alternate (guardian, gdelt, google_news); publishers alternate within each."""
    items = [
        via("n1", "google_news", "P"), via("n2", "google_news", "P"), via("n3", "google_news", "Q"),
        via("d1", "gdelt", "R"),
        via("g1", "guardian", "The Guardian"), via("g2", "guardian", "The Guardian"),
    ]
    order = [c.title for c in interleave_by_source(items)]
    assert order == ["g1", "d1", "n1", "g2", "n3", "n2"]


def test_interleave_includes_gnews() -> None:
    """GNews takes its turn in the source round-robin (after gdelt, before google_news)."""
    items = [via("n1", "google_news", "P"), via("k1", "gnews", "K"), via("k2", "gnews", "L"),
             via("d1", "gdelt", "R"), via("g1", "guardian", "The Guardian")]
    assert [c.title for c in interleave_by_source(items)] == ["g1", "d1", "k1", "n1", "k2"]


def test_gnews_candidates_are_fetched_not_taken_from_content() -> None:
    """GNews results have no full text, so the page is fetched and extracted."""

    class FakeFetcher:
        """Records requested URLs and serves the saved article page."""

        def __init__(self) -> None:
            """Start with no requests."""
            self.urls: list[str] = []

        def fetch_html(self, url: str) -> str:
            """Return the saved article HTML."""
            self.urls.append(url)
            return read_fixture("article.html")

    candidate = parse_gnews_response(json.loads(read_fixture("gnews.json")))[0]
    fetcher = FakeFetcher()
    articles, _ = collect_articles([candidate], "electric vehicles", limit=1, days=3650,
                                   fetcher=fetcher, sources={})  # type: ignore[arg-type]
    assert fetcher.urls == ["https://www.example-energy.com/news/v2g-trial"]  # utm_* removed
    assert articles[0].discovered_via == "gnews"
    assert articles[0].text_type == "full_text"
    assert "twelve fast chargers" in (articles[0].excerpt or "")


def test_every_source_with_relevant_articles_is_represented() -> None:
    """If a source's first candidates are irrelevant, a later relevant one still gets in."""
    body = "Electric vehicles were the focus of a long report published this week. " * 5
    off_topic = "A long report about football results and the weather this week. " * 5
    candidates = interleave_by_source(
        [via(f"news {i}", "google_news", f"Pub{i}", full_text=body) for i in range(5)]
        + [via("guardian off 1", "guardian", "The Guardian", full_text=off_topic),
           via("guardian off 2", "guardian", "The Guardian", full_text=off_topic),
           via("guardian on", "guardian", "The Guardian", full_text=body)]
    )
    articles, _ = collect_articles(candidates, "electric vehicles", limit=2, days=7,
                                   fetcher=None, sources={})  # type: ignore[arg-type]
    assert len(articles) == 2
    assert {a.discovered_via for a in articles} == {"google_news", "guardian"}
