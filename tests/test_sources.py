"""Tests for source parsing, using saved fixtures only (no network)."""

from __future__ import annotations

import json
import logging

import pytest
import requests

from newspulse.config import Settings
from newspulse.sources import gdelt, google_news
from newspulse.sources.base import Candidate, NewsSource, SourceUnavailable, safe_search
from newspulse.sources.gdelt import GdeltSource, build_query, parse_gdelt_response, parse_seendate
from newspulse.sources.google_news import (
    GoogleNewsSource,
    build_feed_url,
    decode_google_news_url,
    parse_google_news_feed,
    strip_publisher_suffix,
)
from newspulse.models import parse_iso
from newspulse.sources.gnews import GNewsSource, parse_gnews_response
from newspulse.sources.gnews import build_query as gnews_build_query
from newspulse.sources.guardian import GuardianSource, parse_guardian_response, strip_html
from tests.conftest import FakeResponse, FakeSession, read_fixture

# ---------------------------------------------------------------- GDELT


def test_gdelt_fixture_is_parsed() -> None:
    """Valid entries become candidates; entries without a URL are skipped."""
    candidates = parse_gdelt_response(read_fixture("gdelt.json"))
    assert len(candidates) == 3
    first = candidates[0]
    assert first.title == "Battery plant for electric vehicles opens in Ohio"
    assert first.source == "example.com"
    assert first.published_at == "2026-11-11T08:15:00Z"
    assert first.discovered_via == "gdelt"
    assert first.full_text is None
    assert candidates[1].title == "EV charging network expands along interstate highways"


def test_gdelt_seendate_conversion() -> None:
    """seendate becomes ISO 8601 UTC; bad values become None."""
    assert parse_seendate("20261112T103000Z") == "2026-11-12T10:30:00Z"
    assert parse_seendate("not a date") is None
    assert parse_seendate(None) is None


def test_gdelt_query_quotes_multi_word_topics() -> None:
    """Multi-word topics are quoted as a phrase; single words are not."""
    assert build_query("electric vehicles") == '"electric vehicles" sourcelang:english'
    assert build_query("cybersecurity") == "cybersecurity sourcelang:english"


def test_gdelt_plain_text_reply_is_an_error_not_a_crash(settings: Settings) -> None:
    """GDELT's plain-text rate-limit message is reported as a source error."""
    session = FakeSession(FakeResponse("Please limit requests to one every 5 seconds."))
    candidates, error = safe_search(GdeltSource(session, settings), "electric vehicles", 7)
    assert candidates == []
    assert error is not None and "non-JSON" in error


def test_gdelt_retries_once_after_rate_limit(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """A 429 triggers exactly one polite retry, then a clear error."""
    monkeypatch.setattr(gdelt.time, "sleep", lambda seconds: None)
    session = FakeSession(FakeResponse("Please limit requests to one every 5 seconds.", status_code=429))
    candidates, error = safe_search(GdeltSource(session, settings), "electric vehicles", 7)
    assert candidates == []
    assert len(session.calls) == 2
    assert error is not None and "rate limit" in error


def test_gdelt_sends_exactly_one_request_on_success(settings: Settings) -> None:
    """A normal run makes a single GDELT call."""
    session = FakeSession(FakeResponse(read_fixture("gdelt.json")))
    GdeltSource(session, settings).search("cybersecurity", 7)
    assert len(session.calls) == 1


def test_gdelt_429_body_is_logged_and_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, settings: Settings
) -> None:
    """The source error is one short sentence; GDELT's full message goes to DEBUG only."""
    monkeypatch.setattr(gdelt.time, "sleep", lambda seconds: None)
    message = "Please limit requests to one every 5 seconds or contact us for larger queries."
    session = FakeSession(FakeResponse(message, status_code=429))
    with caplog.at_level(logging.DEBUG, logger="newspulse.sources.gdelt"):
        _, error = safe_search(GdeltSource(session, settings), "cybersecurity", 7)
    assert error == "GDELT rate limit reached (HTTP 429)"
    body_logs = [r for r in caplog.records if "429 response body" in r.getMessage()]
    assert body_logs and all(r.levelno == logging.DEBUG for r in body_logs)
    assert message in body_logs[0].getMessage()  # full body, not truncated
    assert sum("GDELT request" in r.getMessage() for r in caplog.records) == 2


def test_gdelt_non_json_error_is_short(caplog: pytest.LogCaptureFixture) -> None:
    """A plain-text reply gives a one-sentence error; the body is logged at DEBUG."""
    with caplog.at_level(logging.DEBUG, logger="newspulse.sources.gdelt"):
        with pytest.raises(SourceUnavailable) as info:
            parse_gdelt_response("Some long plain-text error page from the server.")
    assert str(info.value) == "GDELT returned a non-JSON response"
    assert any("plain-text error page" in r.getMessage() and r.levelno == logging.DEBUG
               for r in caplog.records)


def test_long_source_errors_are_cut_at_a_word_boundary(settings: Settings) -> None:
    """Unexpected long errors become one line, cut between words with '...'."""

    class Wordy(NewsSource):
        name = "wordy"

        def search(self, topic: str, days: int) -> list[Candidate]:
            raise RuntimeError("Something went wrong " + "with a very long explanation " * 10
                               + "\nsecond line with a stack trace")

    _, error = safe_search(Wordy(FakeSession(FakeResponse()), settings), "x", 7)
    assert error is not None
    assert len(error) <= 123 and error.endswith("...")
    assert "second line" not in error
    assert error[:-3].split()[-1] in {"Something", "went", "wrong", "with", "a", "very", "long",
                                      "explanation"}


def test_gdelt_session_has_no_automatic_retries(settings: Settings) -> None:
    """build_sources gives GDELT a session whose adapter never retries on its own."""
    from newspulse.sources import build_sources

    shared = requests.Session()
    gdelt_source = next(s for s in build_sources(shared, settings) if s.name == "gdelt")
    assert gdelt_source.session is not shared
    assert gdelt_source.session.get_adapter("https://api.gdeltproject.org").max_retries.total == 0


def test_timeout_gives_readable_error(settings: Settings) -> None:
    """Network timeouts are reported as a short message, not a urllib3 dump."""

    class TimeoutSession(FakeSession):
        def get(self, url: str, params: dict | None = None, timeout: object = None) -> FakeResponse:
            raise requests.ReadTimeout("HTTPSConnectionPool(...): Read timed out.")

    _, error = safe_search(GdeltSource(TimeoutSession(FakeResponse()), settings), "x", 7)
    assert error == "request timed out"


def test_gdelt_empty_body_means_no_results() -> None:
    """An empty body is treated as zero results."""
    assert parse_gdelt_response("") == []
    assert parse_gdelt_response("{}") == []


def test_gdelt_search_sends_expected_params(settings: Settings) -> None:
    """The request uses the documented ArtList parameters."""
    session = FakeSession(FakeResponse(read_fixture("gdelt.json")))
    GdeltSource(session, settings).search("electric vehicles", 3)
    _, params = session.calls[0]
    assert params["mode"] == "ArtList"
    assert params["format"] == "json"
    assert params["timespan"] == "3d"
    assert params["sort"] == "DateDesc"


# ------------------------------------------------------------- Guardian


def test_guardian_fixture_is_parsed() -> None:
    """Guardian results carry full text and an HTML-free snippet."""
    candidates = parse_guardian_response(json.loads(read_fixture("guardian.json")))
    assert len(candidates) == 2
    first = candidates[0]
    assert first.source == "The Guardian"
    assert first.published_at == "2026-11-11T09:45:12Z"
    assert first.full_text and first.full_text.startswith("The sample city council")
    assert first.snippet == "The council says the new electric buses will cut emissions & noise."
    assert candidates[1].full_text is None


def test_guardian_without_key_is_skipped(settings: Settings) -> None:
    """No key -> no request, and the reason is reported."""
    session = FakeSession(FakeResponse("{}"))
    candidates, error = safe_search(GuardianSource(session, settings), "electric vehicles", 7)
    assert candidates == []
    assert error == "GUARDIAN_API_KEY not set"
    assert session.calls == []


def test_guardian_search_with_key(settings: Settings) -> None:
    """With a key, the fixture response is parsed and documented params are sent."""
    keyed = Settings(guardian_api_key="test-key", per_domain_delay=0.0)
    session = FakeSession(FakeResponse(read_fixture("guardian.json")))
    candidates = GuardianSource(session, keyed).search("electric vehicles", 7)
    assert len(candidates) == 2
    _, params = session.calls[0]
    assert params["type"] == "article"  # excludes live blogs and galleries
    assert params["show-fields"] == "bodyText,trailText"
    assert params["order-by"] == "newest"
    assert params["api-key"] == "test-key"


def test_strip_html() -> None:
    """Tags and entities are removed."""
    assert strip_html("<p>A &amp; <b>B</b></p>") == "A & B"
    assert strip_html("") is None


# ---------------------------------------------------------------- GNews


def test_gnews_fixture_is_parsed() -> None:
    """source.name is the publisher, description is the snippet, content is NOT full text."""
    candidates = parse_gnews_response(json.loads(read_fixture("gnews.json")))
    assert len(candidates) == 2  # the untitled entry is skipped
    first = candidates[0]
    assert first.title == "Sample utility trials vehicle-to-grid charging for electric vehicles"
    assert first.source == "Example Energy News"
    assert first.published_at == "2026-11-11T16:20:00Z"
    assert first.snippet == ("A sample utility says electric vehicles parked overnight "
                             "could feed power back to the grid.")
    assert first.full_text is None
    assert first.discovered_via == "gnews"
    assert candidates[1].title == "Electric vehicles charging prices fall in sample region"
    assert candidates[1].snippet is None


def test_gnews_without_key_is_skipped() -> None:
    """No key -> no request, and the reason is reported like the Guardian."""
    session = FakeSession(FakeResponse("{}"))
    candidates, error = safe_search(GNewsSource(session, Settings()), "electric vehicles", 7)
    assert candidates == []
    assert error == "GNEWS_API_KEY not set"
    assert session.calls == []


def test_gnews_search_sends_expected_params() -> None:
    """Documented params are sent; multi-word topics are quoted; `from` is ISO 8601 UTC."""
    session = FakeSession(FakeResponse(read_fixture("gnews.json")))
    candidates = GNewsSource(session, Settings(gnews_api_key="test-key")).search("electric vehicles", 3)
    assert len(candidates) == 2
    url, params = session.calls[0]
    assert url == "https://gnews.io/api/v4/search"
    assert params["q"] == '"electric vehicles"'
    assert params["lang"] == "en"
    assert params["max"] == "10"
    assert params["sortby"] == "publishedAt"
    assert params["apikey"] == "test-key"
    assert params["from"].endswith("Z") and parse_iso(params["from"]) is not None


def test_gnews_single_word_topic_is_not_quoted() -> None:
    """Single words are sent as-is."""
    assert gnews_build_query("cybersecurity") == "cybersecurity"
    assert gnews_build_query('AI "regulation"') == '"AI regulation"'


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "GNews rejected the API key (HTTP 401)"),
        (403, "GNews rejected the API key (HTTP 403)"),
        (429, "GNews daily request limit reached (HTTP 429)"),
        (500, "GNews returned HTTP 500"),
    ],
)
def test_gnews_http_errors_give_short_messages(status: int, expected: str) -> None:
    """Bad key, daily limit and server errors become clear source errors."""
    session = FakeSession(FakeResponse('{"errors": ["nope"]}', status_code=status))
    candidates, error = safe_search(GNewsSource(session, Settings(gnews_api_key="k")), "x", 7)
    assert candidates == []
    assert error == expected


def test_gnews_network_error_never_crashes() -> None:
    """Connection problems are reported, not raised."""

    class DownSession(FakeSession):
        def get(self, url: str, params: dict | None = None, timeout: object = None) -> FakeResponse:
            raise requests.ConnectionError("Failed to establish a new connection")

    _, error = safe_search(GNewsSource(DownSession(FakeResponse()), Settings(gnews_api_key="k")), "x", 7)
    assert error == "could not connect (network error or too many retries)"


def test_gnews_session_has_no_automatic_retries(settings: Settings) -> None:
    """Retries would spend the small daily quota, so GNews gets a no-retry session."""
    from newspulse.sources import build_sources

    shared = requests.Session()
    gnews_source = next(s for s in build_sources(shared, settings) if s.name == "gnews")
    assert gnews_source.session is not shared
    assert gnews_source.session.get_adapter("https://gnews.io").max_retries.total == 0


def test_all_four_sources_are_registered(settings: Settings) -> None:
    """build_sources includes GNews alongside the original three."""
    from newspulse.sources import build_sources

    names = [s.name for s in build_sources(requests.Session(), settings)]
    assert names == ["gdelt", "guardian", "gnews", "google_news"]


# ---------------------------------------------------------- Google News


def test_google_news_fixture_is_parsed() -> None:
    """Publisher comes from <source>, and the ' - Publisher' suffix is removed."""
    candidates = parse_google_news_feed(read_fixture("google_news.xml"))
    assert len(candidates) == 3
    first = candidates[0]
    assert first.title == "Automaker announces cheaper electric vehicles for 2027"
    assert first.source == "Example Times"
    assert first.published_at == "2026-11-11T14:30:00Z"
    assert first.url.startswith("https://news.google.com/rss/articles/")
    assert first.snippet is None  # description only repeated the headline
    assert candidates[2].title == "Regulators review charging standards for electric vehicles"
    assert candidates[2].snippet and "incompatible connectors" in candidates[2].snippet


def test_google_news_feed_url() -> None:
    """The feed URL encodes the topic and the day range."""
    url = build_feed_url("electric vehicles", 7)
    assert url.startswith("https://news.google.com/rss/search?q=electric+vehicles+when:7d")
    assert "ceid=US:en" in url


def test_strip_publisher_suffix() -> None:
    """Only an exact trailing publisher suffix is removed."""
    assert strip_publisher_suffix("Headline - Pub", "Pub") == "Headline"
    assert strip_publisher_suffix("Headline - Other", "Pub") == "Headline - Other"


def test_google_news_decoding_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful decode returns the publisher URL."""
    monkeypatch.setattr(google_news, "gnewsdecoder",
                        lambda url, interval=None, timeout=None: {"success": True,
                                                                  "decoded_url": "https://example.com/a"})
    assert decode_google_news_url("https://news.google.com/rss/articles/X") == "https://example.com/a"


def test_google_news_decoding_failure_keeps_google_link(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed or crashing decode keeps the original Google link."""
    link = "https://news.google.com/rss/articles/X"
    monkeypatch.setattr(google_news, "gnewsdecoder",
                        lambda url, interval=None, timeout=None: {"success": False, "message": "nope"})
    assert decode_google_news_url(link) == link

    def boom(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("network down")

    monkeypatch.setattr(google_news, "gnewsdecoder", boom)
    assert decode_google_news_url(link) == link


def test_google_news_search_does_not_decode(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """search() never decodes links; decoding happens only for candidates actually used."""
    calls: list[str] = []
    monkeypatch.setattr(google_news, "gnewsdecoder",
                        lambda url, **kw: calls.append(url) or {"success": False, "message": "x"})
    session = FakeSession(FakeResponse(read_fixture("google_news.xml")))
    source = GoogleNewsSource(session, settings)
    candidates = source.search("electric vehicles", 7)
    assert calls == []
    source.resolve_url(candidates[0])
    assert len(calls) == 1


# ----------------------------------------------------------- safe_search


class _ExplodingSource(NewsSource):
    """A source that always raises."""

    name = "exploding"

    def search(self, topic: str, days: int) -> list[Candidate]:
        """Always fail."""
        raise ConnectionError("connection refused")


def test_safe_search_catches_everything(settings: Settings) -> None:
    """One failing source returns an error message instead of raising."""
    candidates, error = safe_search(_ExplodingSource(FakeSession(FakeResponse()), settings), "x", 7)
    assert candidates == []
    assert error == "connection refused"


def test_source_unavailable_is_an_exception() -> None:
    """SourceUnavailable can be raised and caught like any exception."""
    with pytest.raises(SourceUnavailable):
        parse_gdelt_response("<html>error</html>")
