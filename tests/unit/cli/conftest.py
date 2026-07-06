"""Shared fixtures for typer-based CLI resource tests.

Each test points the CLI at an ``httpx.MockTransport`` so we can assert on
method, path, query, and body without opening a real socket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from aios.cli.runtime import CliState
from aios_sdk import Client as SdkClient


@dataclass
class Captured:
    """Record of the most recent HTTP call made via the mocked client."""

    method: str = ""
    path: str = ""
    query: dict[str, list[str]] = None  # type: ignore[assignment]
    body: Any = None
    headers: dict[str, str] = None  # type: ignore[assignment]

    def reset(self) -> None:
        self.method = ""
        self.path = ""
        self.query = {}
        self.body = None
        self.headers = {}


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
    """Monkey-patch ``CliState.sdk_client`` to return an SDK ``Client`` with a MockTransport.

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
        captured.headers = dict(request.headers)
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

    def _sdk_client(self: CliState) -> SdkClient:
        client = SdkClient(base_url=self.base_url, token=self.api_key or "")
        client.set_httpx_client(
            httpx.Client(base_url=self.base_url, transport=httpx.MockTransport(handler))
        )
        return client

    monkeypatch.setattr(CliState, "sdk_client", _sdk_client)
    monkeypatch.setenv("AIOS_API_KEY", "test-key")
    monkeypatch.setenv("AIOS_URL", "http://test.invalid")
    return MockedCli(captured=captured, response_queue=responses)


# Minimally-complete response payloads for SDK-backed CLI tests. The SDK's
# ``<Model>.from_dict`` is strict about required fields, so tests asserting
# only request shape still need a complete response to keep the parser from
# raising before the exit-code assertion runs.

_FIXED_TS = "2024-01-01T00:00:00+00:00"

_RESOURCE_BASES: dict[str, dict[str, Any]] = {
    "connection": {
        "id": "conn_01",
        "connector": "signal",
        "external_account_id": "acct-1",
        "metadata": {},
        "created_at": _FIXED_TS,
        "updated_at": _FIXED_TS,
    },
    "vault": {
        "id": "vlt_1",
        "display_name": "test",
        "metadata": {},
        "created_at": _FIXED_TS,
        "updated_at": _FIXED_TS,
    },
    "vault_credential": {
        "id": "cred_1",
        "vault_id": "vlt_1",
        "display_name": "test-cred",
        "target_url": "http://example.invalid",
        "auth_type": "bearer_header",
        "metadata": {},
        "created_at": _FIXED_TS,
        "updated_at": _FIXED_TS,
    },
}


def resource_response(kind: str, **overrides: Any) -> dict[str, Any]:
    """Return a minimally-valid SDK response payload for the given resource kind."""
    return {**_RESOURCE_BASES[kind], **overrides}
