"""Smoke checks for the typed Python SDK.

Verifies the curated public surface imports cleanly, the generated
``Client`` constructs, and a representative operation can be invoked
against an httpx ``MockTransport``. The deeper coverage lives in the
generated tree itself (openapi-python-client emits typed signatures
that mypy already validates) — this file is the boundary check.
"""

from __future__ import annotations

import json

import httpx
import pytest


def test_curated_public_surface_imports() -> None:
    from aios_sdk import (
        Client,
        SseMessage,
        UnexpectedStatus,
        client_from_env,
        parse_sse_lines,
        stream_session,
    )

    assert Client.__name__ == "AuthenticatedClient"
    assert callable(client_from_env)
    assert callable(stream_session)
    assert callable(parse_sse_lines)
    assert SseMessage(event="x", data="y").event == "x"
    assert issubclass(UnexpectedStatus, Exception)


def test_client_constructs_with_base_url_and_token() -> None:
    from aios_sdk import Client

    client = Client(base_url="http://example.test", token="t")
    assert client._base_url == "http://example.test"
    assert client.token == "t"


def test_get_health_operation_against_mocked_transport() -> None:
    from aios_sdk import Client
    from aios_sdk._generated.api.default import get_health

    payload = {"status": "ok", "version": "0.1.0"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json=payload)

    client = Client(base_url="http://test", token="t")
    client.set_httpx_client(
        httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    )
    response = get_health.sync_detailed(client=client)
    assert response.status_code == 200
    assert json.loads(response.content) == payload


def test_client_from_env_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from aios_sdk import client_from_env

    monkeypatch.delenv("AIOS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AIOS_API_KEY"):
        client_from_env()


def test_client_from_env_succeeds_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from aios_sdk import client_from_env

    monkeypatch.setenv("AIOS_API_KEY", "test-key")
    monkeypatch.setenv("AIOS_URL", "http://example.test")
    client = client_from_env()
    assert client._base_url == "http://example.test"
    assert client.token == "test-key"
