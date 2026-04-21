"""Shared fixtures for typer-based CLI resource tests.

Each test points the CLI at an ``httpx.MockTransport`` so we can assert on
method, path, query, and body without opening a real socket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from aios.cli.client import AiosClient
from aios.cli.runtime import CliState


@dataclass
class Captured:
    """Record of the most recent HTTP call made via the mocked client."""

    method: str = ""
    path: str = ""
    query: dict[str, list[str]] = None  # type: ignore[assignment]
    body: Any = None

    def reset(self) -> None:
        self.method = ""
        self.path = ""
        self.query = {}
        self.body = None


@dataclass
class MockedCli:
    """Handle returned by the ``mocked_cli`` fixture.

    Callers queue responses via ``queue_response(httpx.Response(...))``; each
    request pops the next queued response. After a call, ``captured`` has
    the request shape (method/path/query/body).
    """

    captured: Captured
    response_queue: list[httpx.Response]

    def queue_response(self, response: httpx.Response) -> None:
        self.response_queue.append(response)


@pytest.fixture
def mocked_cli(monkeypatch: pytest.MonkeyPatch) -> MockedCli:
    """Monkey-patch ``CliState.client`` to return an AiosClient with a MockTransport.

    Tests call ``handle.queue_response(httpx.Response(...))`` before invoking
    a command; each request pops the next queued response (or returns 500
    if the queue is empty).
    """
    captured = Captured()
    captured.reset()
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.method = request.method
        captured.path = request.url.path
        captured.query = {k: request.url.params.get_list(k) for k in request.url.params}
        if request.content:
            import json

            try:
                captured.body = json.loads(request.content)
            except Exception:
                captured.body = request.content
        else:
            captured.body = None
        if not responses:
            return httpx.Response(500, json={"error": "no response queued"})
        return responses.pop(0)

    def _client(self: CliState) -> AiosClient:  # type: ignore[override]
        return AiosClient(
            base_url=self.base_url,
            api_key=self.api_key,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(CliState, "client", _client)
    monkeypatch.setenv("AIOS_API_KEY", "test-key")
    monkeypatch.setenv("AIOS_URL", "http://test.invalid")
    return MockedCli(captured=captured, response_queue=responses)
