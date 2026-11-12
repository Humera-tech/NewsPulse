"""Tests for the extractive summarizer and Gemini fallback (Gemini is always mocked)."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from newspulse.config import Settings
from newspulse.extractor import extract_main_text
from newspulse.models import Article, utc_now_iso
from newspulse.pipeline import summarize_articles
from newspulse.summarizer import (
    ExtractiveSummarizer,
    GeminiSummarizer,
    build_summarizer,
    split_sentences,
)
from tests.conftest import read_fixture


@pytest.fixture
def article_text() -> str:
    """Main text extracted from the saved article page."""
    text = extract_main_text(read_fixture("article.html"))
    assert text
    return text


def test_extractive_summary_has_2_to_4_sentences_from_source(article_text: str) -> None:
    """Every summary sentence appears word-for-word in the source text."""
    summary, method = ExtractiveSummarizer().summarize("Charging hub opens", article_text)
    assert method == "extractive"
    assert summary
    sentences = split_sentences(summary)
    assert 2 <= len(sentences) <= 4
    for sentence in sentences:
        assert sentence in article_text


def test_extractive_summary_on_guardian_body() -> None:
    """Works on API-provided full text too."""
    body = json.loads(read_fixture("guardian.json"))["response"]["results"][0]["fields"]["bodyText"]
    summary, _ = ExtractiveSummarizer().summarize("Electric buses", body)
    sentences = split_sentences(summary)
    assert 2 <= len(sentences) <= 4
    assert all(s in body for s in sentences)


def test_extractive_summary_keeps_original_order(article_text: str) -> None:
    """Selected sentences are returned in the order they appear in the text."""
    summary, _ = ExtractiveSummarizer().summarize("", article_text)
    positions = [article_text.index(s) for s in split_sentences(summary)]
    assert positions == sorted(positions)


def test_empty_text_gives_no_summary() -> None:
    """Empty text -> (None, None)."""
    assert ExtractiveSummarizer().summarize("t", "") == (None, None)
    assert GeminiSummarizer(api_key=None, model="m").summarize("t", None) == (None, None)


def test_split_sentences_keeps_closing_quotes() -> None:
    """Closing quotes stay attached to their sentence."""
    text = 'He said "it works." Then he left. Prices rose 2.5 percent.'
    assert split_sentences(text) == ['He said "it works."', "Then he left.", "Prices rose 2.5 percent."]


def test_split_does_not_break_model_names() -> None:
    """'ID.UNYX' stays intact (no space after the period)."""
    text = "Volkswagen expanded the ID.UNYX brand in China this year. Sales were strong."
    assert split_sentences(text) == [
        "Volkswagen expanded the ID.UNYX brand in China this year.",
        "Sales were strong.",
    ]


def test_split_does_not_break_on_us() -> None:
    """'U.S.' is not a sentence end, whether followed by lowercase or a capital."""
    text = "U.S. sales rose 10% in October. Officials at the U.S. Department of Energy agreed."
    assert split_sentences(text) == [
        "U.S. sales rose 10% in October.",
        "Officials at the U.S. Department of Energy agreed.",
    ]


def test_split_does_not_break_decimals() -> None:
    """'$4.5 billion' is not split."""
    text = "The company raised $4.5 billion in new funding. Investors welcomed the news."
    assert split_sentences(text) == [
        "The company raised $4.5 billion in new funding.",
        "Investors welcomed the news.",
    ]


def test_split_respects_abbreviations_and_initials() -> None:
    """Dr., Inc., No., the ID. prefix and single initials do not end sentences."""
    text = ("Dr. Jane Smith of Example Inc. said the No. 3 plant opens soon. "
            "J. R. Doe agreed. The ID. Buzz van was shown.")
    assert split_sentences(text) == [
        "Dr. Jane Smith of Example Inc. said the No. 3 plant opens soon.",
        "J. R. Doe agreed.",
        "The ID. Buzz van was shown.",
    ]


def test_split_requires_space_and_capital_after_period() -> None:
    """No split when the next word is lowercase or there is no space."""
    assert split_sentences("Version 2.0 shipped. then more.") == ["Version 2.0 shipped. then more."]


def test_extractive_rejects_fragments() -> None:
    """Short sentences and ones starting in lowercase are never chosen."""
    text = (
        "Short line here. "
        "and this one starts with a lowercase letter but is otherwise long enough to count. "
        "The council approved a new budget for public transport on Monday evening. "
        "Officials said the money would pay for forty new electric buses next year. "
        "Residents had asked for more frequent services during a public consultation."
    )
    summary, _ = ExtractiveSummarizer().summarize("Budget", text)
    sentences = split_sentences(summary)
    assert 2 <= len(sentences) <= 4
    for sentence in sentences:
        assert len(sentence.split()) >= 8
        assert sentence[0].isupper()
    assert "Short line here." not in summary
    assert "and this one" not in summary


class FakeAPIError(Exception):
    """Mimics google.genai.errors.APIError, which carries an HTTP `code`."""

    def __init__(self, code: int, message: str) -> None:
        """Store the status code and message."""
        super().__init__(f"{code} {message}")
        self.code = code


def test_gemini_model_not_found_disables_gemini_once(
    caplog: pytest.LogCaptureFixture, article_text: str
) -> None:
    """A 404 logs one warning and stops calling Gemini for the rest of the run."""
    client = MagicMock()
    client.models.generate_content.side_effect = FakeAPIError(404, "models/gemini-x is not found")
    summarizer = GeminiSummarizer("key", "gemini-x", client=client)
    with caplog.at_level(logging.WARNING, logger="newspulse.summarizer"):
        results = [summarizer.summarize(f"T{i}", article_text) for i in range(3)]
    assert all(method == "extractive" for _, method in results)
    assert client.models.generate_content.call_count == 1
    warnings = [r.getMessage() for r in caplog.records if "is unavailable" in r.getMessage()]
    assert len(warnings) == 1
    assert "Gemini model gemini-x is unavailable" in warnings[0]
    assert "using extractive summaries for this run" in warnings[0]


def test_gemini_auth_error_disables_gemini(article_text: str) -> None:
    """An invalid API key is treated like an unavailable model."""
    client = MagicMock()
    client.models.generate_content.side_effect = FakeAPIError(400, "API key not valid. API_KEY_INVALID")
    summarizer = GeminiSummarizer("bad", "m", client=client)
    summarizer.summarize("A", article_text)
    summarizer.summarize("B", article_text)
    assert client.models.generate_content.call_count == 1


def test_gemini_503_twice_then_success(caplog: pytest.LogCaptureFixture, article_text: str) -> None:
    """Two 503s are retried after 3s and 8s; the third attempt succeeds with no warning."""
    client = MagicMock()
    client.models.generate_content.side_effect = [
        FakeAPIError(503, "UNAVAILABLE"),
        FakeAPIError(503, "UNAVAILABLE"),
        MagicMock(text="A grounded summary."),
    ]
    waits: list[float] = []
    summarizer = GeminiSummarizer("key", "m", client=client, sleep=waits.append)
    with caplog.at_level(logging.DEBUG, logger="newspulse.summarizer"):
        result = summarizer.summarize("T", article_text)
    assert result == ("A grounded summary.", "gemini")
    assert client.models.generate_content.call_count == 3
    assert waits == [3.0, 8.0]
    retry_logs = [r for r in caplog.records if "retrying" in r.getMessage()]
    assert len(retry_logs) == 2 and all(r.levelno == logging.DEBUG for r in retry_logs)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_gemini_transient_errors_exhaust_retries(caplog: pytest.LogCaptureFixture, article_text: str) -> None:
    """429, timeout and 503 on all 3 attempts -> one WARNING, extractive for that article only."""
    client = MagicMock()
    client.models.generate_content.side_effect = [
        FakeAPIError(429, "RESOURCE_EXHAUSTED"),
        TimeoutError("timed out"),
        FakeAPIError(503, "UNAVAILABLE"),
        MagicMock(text="Next article works."),
    ]
    summarizer = GeminiSummarizer("key", "m", client=client, sleep=no_sleep)
    with caplog.at_level(logging.DEBUG, logger="newspulse.summarizer"):
        first = summarizer.summarize("A", article_text)
        second = summarizer.summarize("B", article_text)
    assert first[1] == "extractive"
    assert second == ("Next article works.", "gemini")  # Gemini is still used afterwards
    assert client.models.generate_content.call_count == 4
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "after 3 attempts" in warnings[0]


def test_gemini_quota_429_disables_gemini_once(
    caplog: pytest.LogCaptureFixture, article_text: str
) -> None:
    """A 429 that mentions quota is not retried: one warning, then no more Gemini calls."""
    client = MagicMock()
    client.models.generate_content.side_effect = FakeAPIError(
        429, "RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan.")
    waits: list[float] = []
    summarizer = GeminiSummarizer("key", "m", client=client, sleep=waits.append)
    with caplog.at_level(logging.WARNING, logger="newspulse.summarizer"):
        methods = [summarizer.summarize(f"T{i}", article_text)[1] for i in range(3)]
    assert methods == ["extractive"] * 3
    assert client.models.generate_content.call_count == 1
    assert waits == []
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["Gemini quota exceeded; using extractive summaries for this run"]


def test_gemini_429_without_quota_is_still_retried(article_text: str) -> None:
    """A plain rate-limit 429 (no 'quota') keeps the 3s/8s retries."""
    client = MagicMock()
    client.models.generate_content.side_effect = [
        FakeAPIError(429, "RESOURCE_EXHAUSTED. Too many requests, slow down."),
        MagicMock(text="Recovered summary."),
    ]
    waits: list[float] = []
    result = GeminiSummarizer("key", "m", client=client, sleep=waits.append).summarize("T", article_text)
    assert result == ("Recovered summary.", "gemini")
    assert waits == [3.0]


def test_gemini_client_has_30s_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real client is created with a 30-second (30000 ms) request timeout."""
    from google import genai

    factory = MagicMock()
    monkeypatch.setattr(genai, "Client", factory)
    GeminiSummarizer(api_key="key", model="m")
    http_options = factory.call_args.kwargs["http_options"]
    assert factory.call_args.kwargs["api_key"] == "key"
    assert http_options.timeout == 30_000


def test_gemini_http_timeout_is_retried(article_text: str) -> None:
    """httpx timeouts raised by the SDK get the same 3s/8s retries as 503."""
    import httpx

    client = MagicMock()
    client.models.generate_content.side_effect = [
        httpx.ReadTimeout("The read operation timed out"),
        httpx.ConnectTimeout("connect timed out"),
        MagicMock(text="Third time lucky."),
    ]
    waits: list[float] = []
    result = GeminiSummarizer("key", "m", client=client, sleep=waits.append).summarize("T", article_text)
    assert result == ("Third time lucky.", "gemini")
    assert waits == [3.0, 8.0]


def test_gemini_404_is_not_retried(article_text: str) -> None:
    """Model-not-found stops immediately, without retry waits."""
    client = MagicMock()
    client.models.generate_content.side_effect = FakeAPIError(404, "model not found")
    waits: list[float] = []
    GeminiSummarizer("key", "m", client=client, sleep=waits.append).summarize("T", article_text)
    assert client.models.generate_content.call_count == 1
    assert waits == []


def test_gemini_without_key_falls_back(monkeypatch: pytest.MonkeyPatch, article_text: str) -> None:
    """With no key, Gemini is never called and the extractive method is recorded."""
    from google import genai

    client_factory = MagicMock(side_effect=AssertionError("client must not be created"))
    monkeypatch.setattr(genai, "Client", client_factory)
    summarizer = build_summarizer("gemini", Settings(gemini_api_key=None))
    summary, method = summarizer.summarize("Charging hub opens", article_text)
    assert method == "extractive"
    assert summary
    client_factory.assert_not_called()


def no_sleep(seconds: float) -> None:
    """Stand-in for time.sleep so retry tests run instantly."""


def test_gemini_error_falls_back(article_text: str) -> None:
    """A non-retryable Gemini error falls back to extractive immediately."""
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("500 internal error")
    summarizer = GeminiSummarizer("key", "gemini-3.8-flash", client=client, sleep=no_sleep)
    summary, method = summarizer.summarize("T", article_text)
    assert method == "extractive"
    assert summary
    assert client.models.generate_content.call_count == 1


def test_gemini_empty_response_falls_back(article_text: str) -> None:
    """An empty Gemini reply is treated as a failure."""
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="")
    _, method = GeminiSummarizer("key", "m", client=client).summarize("T", article_text)
    assert method == "extractive"


def test_gemini_success_uses_only_collected_text(article_text: str) -> None:
    """Gemini gets the configured model, low temperature and only the given title/text."""
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="  A grounded summary.  ")
    summary, method = GeminiSummarizer("key", "gemini-2.5-flash", client=client).summarize(
        "Charging hub opens", article_text)
    assert (summary, method) == ("A grounded summary.", "gemini")
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["config"] == {"temperature": 0.2}
    assert "Use ONLY facts stated" in kwargs["contents"]
    assert article_text in kwargs["contents"]


def test_summarize_articles_flags_missing_summary() -> None:
    """Articles without text get summary=None and 'summary' in missing_fields."""
    article = Article(title="T", source="S", url="https://e.example/a", published_at=None,
                      retrieved_at=utc_now_iso(), discovered_via="gdelt", excerpt=None, text_type="none")
    summarize_articles([article], ExtractiveSummarizer())
    assert article.summary is None
    assert article.summary_method is None
    assert article.missing_fields == ["published_at", "excerpt", "summary"]
