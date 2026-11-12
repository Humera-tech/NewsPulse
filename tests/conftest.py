"""Shared pytest fixtures: saved sample responses and a fake HTTP session."""

from __future__ import annotations

from pathlib import Path

import pytest

from newspulse.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, text: str = "", status_code: int = 200, headers: dict | None = None) -> None:
        """Store the body, status and headers."""
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status_code
        self.headers = headers or {}
        self.encoding = "utf-8"

    def json(self) -> object:
        """Parse the body as JSON like requests does."""
        import json

        return json.loads(self.text)


class FakeSession:
    """Records requests and returns a pre-set response; never touches the network."""

    def __init__(self, response: FakeResponse) -> None:
        """Store the canned response."""
        self.response = response
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> FakeResponse:
        """Record the call and return the canned response."""
        self.calls.append((url, params))
        return self.response


def read_fixture(name: str) -> str:
    """Return the text of a saved fixture file."""
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def settings() -> Settings:
    """Settings with no API keys and no delays."""
    return Settings(gemini_api_key=None, guardian_api_key=None, per_domain_delay=0.0)
