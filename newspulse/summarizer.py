"""Summarizers: Gemini (summaries only) and an extractive, non-AI fallback.

Gemini is used ONLY here, and only on text the code has already collected.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from newspulse.config import Settings
from newspulse.filters import STOPWORDS
from newspulse.models import shorten_message

logger = logging.getLogger(__name__)

PROMPT = (
    "Summarize the following news article in 2 to 4 sentences. Use ONLY facts stated "
    "in the text below. Do not add background, opinions or anything not in the text. If the "
    "text is too short to summarize, summarize only what is there. Return only the summary."
)

SummaryResult = tuple[str | None, str | None]

# Waits before the 2nd and 3rd Gemini attempt on 503 / 429 / timeout.
RETRY_DELAYS: tuple[float, ...] = (3.0, 8.0)
GEMINI_TIMEOUT_SECONDS = 30.0  # per Gemini request; without it one stuck call can block the run

# A candidate boundary: . ! or ? (plus optional closing quotes/brackets), whitespace,
# then a capital letter, digit or opening quote. Abbreviations are filtered out later.
_BOUNDARY_RE = re.compile(r"[.!?][\"'”’)\]]*(\s+)(?=[\"'“‘(\[]?[A-Z0-9])")
_TOKEN_BEFORE_PERIOD_RE = re.compile(r"(\S+)\.$")
_DOTTED_ACRONYM_RE = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")  # U.S, U.K, e.g, i.e
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)\]]*$")
_SENTENCE_START_RE = re.compile(r"^[A-Z0-9\"'“‘]")
_WORD_RE = re.compile(r"[a-z0-9']+")

ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sr jr st mt ft ave rd gen col lt sgt capt cmdr adm gov sen rep
    rev hon pres inc ltd co corp llc plc bros dept univ est approx vs etc no nos vol
    fig pp ed eds op id jan feb mar apr jun jul aug sep sept oct nov dec
    """.split()
)

MIN_SENTENCE_CHARS = 40
MIN_SENTENCE_WORDS = 8


class Summarizer(Protocol):
    """Anything that can summarize an article's text."""

    def summarize(self, title: str, text: str | None) -> SummaryResult:
        """Return (summary, method), or (None, None) if there is no text."""


def _ends_with_abbreviation(text_before: str) -> bool:
    """Return True if text ending in '.' ends with an abbreviation, initial or acronym."""
    match = _TOKEN_BEFORE_PERIOD_RE.search(text_before)
    if not match:
        return False
    token = match.group(1).lstrip("\"'(“‘[")
    return (
        token.lower() in ABBREVIATIONS
        or (len(token) == 1 and token.isupper())
        or bool(_DOTTED_ACRONYM_RE.fullmatch(token))
    )


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; every sentence is an exact substring of the text.

    Does not split inside abbreviations ("Dr.", "Inc.", "ID.", "U.S."), single
    initials, decimals ("$4.5"), or where no space + capital/quote follows.
    """
    sentences: list[str] = []
    for paragraph in text.splitlines():
        start = 0
        for match in _BOUNDARY_RE.finditer(paragraph):
            punct_end = match.start(1)
            plain_period = paragraph[match.start()] == "." and punct_end == match.start() + 1
            if plain_period and _ends_with_abbreviation(paragraph[start:punct_end]):
                continue
            piece = paragraph[start:punct_end].strip()
            if piece:
                sentences.append(piece)
            start = match.end(1)
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def is_full_sentence(sentence: str) -> bool:
    """Reject fragments: need 8+ words, a capital/digit/quote start and end punctuation."""
    return (
        len(sentence) >= MIN_SENTENCE_CHARS
        and len(sentence.split()) >= MIN_SENTENCE_WORDS
        and bool(_SENTENCE_START_RE.match(sentence))
        and bool(_SENTENCE_END_RE.search(sentence))
    )


def _content_words(sentence: str) -> list[str]:
    """Return lowercase non-stopword words of a sentence."""
    return [w for w in _WORD_RE.findall(sentence.lower()) if w not in STOPWORDS and len(w) > 1]


def _target_count(n_sentences: int) -> int:
    """Choose how many sentences (2-4) to keep for a text of this length."""
    if n_sentences <= 6:
        return 2
    if n_sentences <= 15:
        return 3
    return 4


class ExtractiveSummarizer:
    """Word-frequency extractive summarizer. Uses no AI; never rewrites sentences."""

    def summarize(self, title: str, text: str | None) -> SummaryResult:
        """Pick the 2-4 highest-scoring full sentences, in original order.

        Returns (None, None) when the text has no complete sentence at all.
        """
        if not text or not text.strip():
            return None, None
        sentences = split_sentences(text)
        candidates = [s for s in sentences if is_full_sentence(s)]
        if not candidates:
            return None, None  # never return text that fails the sentence rules
        if len(candidates) <= 2:
            return " ".join(candidates), "extractive"

        frequencies = Counter(w for s in candidates for w in _content_words(s))
        top = max(frequencies.values(), default=1)
        title_words = set(_content_words(title or ""))

        def score(sentence: str) -> float:
            """Average normalized word frequency, with a small bonus for title words."""
            words = _content_words(sentence)
            if not words:
                return 0.0
            base = sum(frequencies[w] / top for w in words) / len(words)
            return base + 0.1 * len(title_words.intersection(words))

        count = min(_target_count(len(candidates)), len(candidates))
        ranked = sorted(range(len(candidates)), key=lambda i: score(candidates[i]), reverse=True)
        return " ".join(candidates[i] for i in sorted(ranked[:count])), "extractive"


def build_prompt(title: str, text: str) -> str:
    """Build the Gemini prompt containing only the instructions, title and collected text."""
    return f"{PROMPT}\n\nTitle: {title}\n\nText:\n{text}"


def is_quota_error(exc: BaseException) -> bool:
    """Return True for a 429 / RESOURCE_EXHAUSTED that says the quota is used up."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = str(exc).lower()
    is_429 = code == 429 or "429" in message or "resource_exhausted" in message
    return is_429 and "quota" in message


def is_retryable_gemini_error(exc: BaseException) -> bool:
    """Return True for temporary errors worth retrying: 503 UNAVAILABLE, 429 or a timeout."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 503):
        return True
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return True
    message = str(exc).lower()
    markers = ("429", "503", "resource_exhausted", "unavailable", "timed out", "deadline exceeded")
    return any(marker in message for marker in markers)


def is_fatal_gemini_error(exc: BaseException) -> bool:
    """Return True for errors that will repeat on every call: model not found or bad auth."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = str(exc).lower()
    if code in (401, 403, 404):
        return True
    if code == 400 and ("api key" in message or "api_key" in message):
        return True
    markers = ("not found", "api_key_invalid", "permission_denied", "unauthenticated", "api key not valid")
    return any(marker in message for marker in markers)


class GeminiSummarizer:
    """Summarizes collected text with Gemini; falls back to extractive on any problem."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        client: Any | None = None,
        fallback: Summarizer | None = None,
        retry_delays: Sequence[float] = RETRY_DELAYS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the Gemini client if a key is available (or use an injected client)."""
        self.model = model
        self.fallback = fallback or ExtractiveSummarizer()
        self.retry_delays = tuple(retry_delays)
        self._sleep = sleep
        self.client = client
        if self.client is None and api_key:
            try:
                from google import genai
                from google.genai import types

                self.client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=int(GEMINI_TIMEOUT_SECONDS * 1000)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create Gemini client (%s); using extractive summaries", exc)
                self.client = None
        if self.client is None:
            logger.warning("GEMINI_API_KEY not set; using extractive summaries")

    def _disable(self, message: str, *args: object) -> None:
        """Stop using Gemini for the rest of the run, warning exactly once."""
        logger.warning(message, *args)
        self.client = None

    def summarize(self, title: str, text: str | None) -> SummaryResult:
        """Return a Gemini summary, or an extractive one if Gemini is unavailable or fails."""
        if not text or not text.strip():
            return None, None
        if self.client is None:
            return self.fallback.summarize(title, text)

        prompt = build_prompt(title, text)
        attempts = len(self.retry_delays) + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._generate(prompt), "gemini"
            except Exception as exc:  # noqa: BLE001 - one Gemini error must not stop the run
                logger.debug("Gemini error detail: %r", exc)
                detail = shorten_message(str(exc), 160) or type(exc).__name__
                if is_fatal_gemini_error(exc):
                    self._disable("Gemini model %s is unavailable: %s; using extractive summaries "
                                  "for this run", self.model, detail)
                    break
                if is_quota_error(exc):
                    # Retrying cannot help until the quota resets, and retries use more of it.
                    self._disable("Gemini quota exceeded; using extractive summaries for this run")
                    break
                if is_retryable_gemini_error(exc) and attempt < attempts:
                    delay = self.retry_delays[attempt - 1]
                    logger.debug("Gemini attempt %d/%d failed for %r (%s); retrying in %.0fs",
                                 attempt, attempts, title[:60], detail, delay)
                    self._sleep(delay)
                    continue
                tries = f" after {attempt} attempts" if attempt > 1 else ""
                logger.warning("Gemini failed for %r%s (%s); using extractive summary",
                               title[:60], tries, detail)
                break
        return self.fallback.summarize(title, text)

    def _generate(self, prompt: str) -> str:
        """Make one Gemini call and return the cleaned summary text (raises on empty)."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": 0.2},
        )
        summary = " ".join((getattr(response, "text", None) or "").split())
        if not summary:
            raise ValueError("empty response")
        return summary


def build_summarizer(name: str, settings: Settings) -> Summarizer:
    """Return the summarizer selected on the command line."""
    if name == "gemini":
        return GeminiSummarizer(settings.gemini_api_key, settings.gemini_model)
    return ExtractiveSummarizer()
