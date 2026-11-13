"""Tests for URL normalization, date/relevance filters and deduplication."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from newspulse.filters import (
    clean_title,
    clean_url,
    dedupe,
    filter_by_date,
    is_academic_url,
    is_finance_spam,
    is_media_url,
    is_relevant,
    normalize_url,
    repair_gdelt_spacing,
)
from newspulse.pipeline import (
    choose_title,
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


def test_single_word_topic_uses_500_char_window() -> None:
    """Single-word topics search the title and the first 500 characters (like multi-word)."""
    assert is_relevant("cybersecurity", "Bank report", "x " * 200 + "cybersecurity spending rose")
    assert not is_relevant("cybersecurity", "Bank report", "x " * 300 + "cybersecurity")
    assert is_relevant("cybersecurity", "Cybersecurity budget grows", "x " * 2000)


# ------------------------------------------------------ finance spam


def test_finance_spam_titles_are_detected() -> None:
    """Ticker-in-parentheses and stock-move phrases are auto-generated posts."""
    for title in (
        "SysGroup (LON:SYS) Stock Price Down 2.8% - Here's Why",
        "SysGroup ( LON : SYS ) Stock Price Down 2 . 8 % – Here Why",  # raw GDELT spacing
        "Example Corp (NASDAQ:EXMP) Shares Sold by Sample Advisors",
        "Sample Inc (NYSE:SMP) Raises Dividend",
        "Maple Co (TSE:MPL) holds steady",
        "Widget Ltd Stock Price Up 4% on Volume",
        "Example Corp Shares Gap Down",
        "Analysts Lift Example Corp Price Target to $40",
    ):
        assert is_finance_spam(title), title


def test_normal_news_is_not_finance_spam() -> None:
    """Real headlines, including ones about markets, are kept."""
    for title in (
        "Global EV sales rise 8% in October as Europe, North America rebound",
        "Stock markets fall after rate decision (live updates)",
        "Richmond Public Schools expands identity monitoring after cybersecurity breach",
        "Company (formerly Example) announces new CEO",
    ):
        assert not is_finance_spam(title), title


def test_prepare_candidates_drops_finance_spam() -> None:
    """Stock-ticker posts never reach the fetch stage."""
    items = [make("SysGroup ( LON : SYS ) Stock Price Down 2 . 8 %", "https://a.example/1",
                  published_at="2026-11-11T00:00:00Z"),
             make("EV makers report record sales", "https://b.example/2",
                  published_at="2026-11-11T00:00:00Z")]
    assert [c.title for c in prepare_candidates(items, days=30)] == ["EV makers report record sales"]


# ------------------------------------------------------- title cleaning


def test_clean_title_removes_breadcrumbs() -> None:
    """Everything from the first ' > ' is dropped."""
    title = "Airmen honored at ceremony > Office of Special Investigations > Article Display"
    assert clean_title(title, None) == "Airmen honored at ceremony"


def test_clean_title_removes_publisher_suffix() -> None:
    """' | X' and ' - X' are removed only when X is the publisher."""
    assert clean_title("EV sales rise | Example Times", "Example Times") == "EV sales rise"
    assert clean_title("EV sales rise - Example Times", "Example Times") == "EV sales rise"
    assert clean_title("EV sales rise - Daily Political", "dailypolitical.com") == "EV sales rise"
    assert clean_title("Q&A - What comes next", "Example Times") == "Q&A - What comes next"
    assert clean_title("Example Times", "Example Times") == "Example Times"  # never empty


def test_repair_gdelt_spacing() -> None:
    """GDELT's spaced-out punctuation is repaired."""
    assert repair_gdelt_spacing("SysGroup ( LON : SYS ) Stock Price Down 2 . 8 %") == \
        "SysGroup (LON:SYS) Stock Price Down 2.8%"
    assert repair_gdelt_spacing("( LON : SYS )") == "(LON:SYS)"
    assert repair_gdelt_spacing("2 . 8 %") == "2.8%"
    assert repair_gdelt_spacing("She earned a Ph . D . in physics") == "She earned a Ph.D. in physics"
    assert repair_gdelt_spacing("Prices rise , sales fall ; analysts : wait !") == \
        "Prices rise, sales fall; analysts: wait!"
    assert repair_gdelt_spacing("Talks in the U . S . end") == "Talks in the U.S. end"


def test_gdelt_titles_are_repaired_in_prepare() -> None:
    """prepare_candidates applies the GDELT repair and title cleaning."""
    item = make("Sample Co ( UK ) wins contract , orders rise 2 . 5 % > News > Article Display",
                "https://c.example/3", published_at="2026-11-11T00:00:00Z")
    assert prepare_candidates([item], days=30)[0].title == \
        "Sample Co (UK) wins contract, orders rise 2.5%"


def test_gdelt_prefers_page_title_when_it_matches() -> None:
    """The page's own title replaces GDELT's, but only if it is the same headline."""
    candidate = make("Scout Motors has 212 000 reservations as plant nears completion",
                     "https://d.example/4", source="cnbc.com")
    page = "Scout Motors has 212,000 reservations as plant nears completion | CNBC"
    assert choose_title(candidate, page) == \
        "Scout Motors has 212,000 reservations as plant nears completion"
    assert choose_title(candidate, "Subscribe to our newsletter") == candidate.title
    assert choose_title(candidate, None) == candidate.title
    other = Candidate(title="Original title", url="https://e.example", source="X", published_at=None,
                      snippet=None, discovered_via="google_news")
    assert choose_title(other, "Original title, edited") == "Original title"  # GDELT only


def test_gdelt_article_uses_page_title_end_to_end() -> None:
    """A fetched GDELT page supplies the title stored on the Article."""

    class PageFetcher:
        """Serves the saved article page."""

        def fetch_html(self, url: str) -> str:
            """Return the saved article HTML."""
            return read_fixture("article.html")

    candidate = make("Sample town opens public charging hub for electric vehicles | Example Gazette",
                     "https://f.example/5", published_at="2026-11-10T12:00:00Z", source="example.com")
    candidate.title = repair_gdelt_spacing("Sample town opens public charging hub for electric vehicles")
    articles, _ = collect_articles([candidate], "electric vehicles", limit=1, days=3650,
                                   fetcher=PageFetcher(), sources={})  # type: ignore[arg-type]
    assert articles[0].title == "Sample town opens public charging hub for electric vehicles"


# ------------------------------------------------- video / podcast / live


def test_media_urls_are_detected() -> None:
    """Video, podcast, gallery and live pages are recognised by path segment."""
    for url in (
        "https://www.foxbusiness.com/video/6408311436112",
        "https://www.example.com/news/videos/ev-launch",
        "https://www.youtube.com/watch?v=abc123",
        "https://www.example.com/podcast/episode-12",
        "https://www.example.com/podcasts/tech-weekly/ep-3",
        "https://www.example.com/gallery/motor-show-in-pictures",
        "https://www.example.com/live-tv",
        "https://www.theguardian.com/business/live/2026/nov/10/ev-motor-show-live",
    ):
        assert is_media_url(url), url


def test_ordinary_articles_are_not_media() -> None:
    """Words that merely contain a pattern do not count."""
    for url in (
        "https://www.example.com/news/watchdog-fines-carmaker",
        "https://www.example.com/business/video-game-sales-rise",
        "https://www.example.com/2026/11/12/living-with-an-ev",
        "https://www.theguardian.com/environment/2026/nov/11/electric-buses",
    ):
        assert not is_media_url(url), url


def test_prepare_candidates_drops_media_pages() -> None:
    """Video and live pages are skipped before fetching."""
    items = [make("Expert discusses EV battery fires", "https://www.foxbusiness.com/video/6408311436112",
                  published_at="2026-11-11T00:00:00Z"),
             make("EV motor show live", "https://www.example.co.uk/business/live/2026/nov/10/show",
                  published_at="2026-11-11T00:00:00Z"),
             make("EV makers report record sales", "https://b.example/2",
                  published_at="2026-11-11T00:00:00Z")]
    assert [c.title for c in prepare_candidates(items, days=30)] == ["EV makers report record sales"]


def test_media_page_found_after_google_decoding_is_skipped() -> None:
    """A Google link that decodes to a video page is skipped without fetching it."""

    class DecodesToVideo:
        """Fake Google News source whose link decodes to a video page."""

        def resolve_url(self, candidate: Candidate) -> str:
            """Return a video URL."""
            return "https://www.foxbusiness.com/video/6408311436112"

    class NoFetch:
        """Fails the test if any page is fetched."""

        def fetch_html(self, url: str) -> str:
            """Never expected to be called."""
            raise AssertionError(f"should not fetch {url}")

    candidate = Candidate(title="Cybersecurity expert breaks down health check failure",
                          url="https://news.google.com/rss/articles/X", source="Fox Business",
                          published_at=None, snippet=None, discovered_via="google_news")
    articles, stats = collect_articles([candidate], "cybersecurity", limit=1, days=7,
                                       fetcher=NoFetch(),  # type: ignore[arg-type]
                                       sources={"google_news": DecodesToVideo()})  # type: ignore[dict-item]
    assert articles == []
    assert stats.media_pages == 1


def test_article_without_usable_text_is_skipped() -> None:
    """A page with junk text and no snippet is dropped, not kept as a title-only filler."""

    class JunkFetcher:
        """Serves a page whose only text is a TV schedule."""

        def fetch_html(self, url: str) -> str:
            """Return a schedule page."""
            rows = "".join(f"<p>{h}:00 AM - {h}:30 AM</p><p>Paid Programming</p>" for h in range(1, 12))
            return f"<html><body><article><h1>Cybersecurity</h1>{rows}</article></body></html>"

    candidate = make("Cybersecurity expert on cyberattack fears", "https://g.example/7",
                     published_at="2026-11-12T00:00:00Z")
    articles, stats = collect_articles([candidate], "cybersecurity", limit=1, days=3650,
                                       fetcher=JunkFetcher(), sources={})  # type: ignore[arg-type]
    assert articles == []
    assert stats.no_text == 1


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
