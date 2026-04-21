"""Tests for :class:`aios.cli.client.AiosClient`.

Uses ``httpx.MockTransport`` so we never open a real socket. Tests cover
auth header injection, error envelope decoding, and SSE streaming.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from aios.cli.client import AiosApiError, AiosClient


def _mock_client(handler, *, api_key: str | None = "key-123") -> AiosClient:
    """Build an AiosClient whose underlying httpx.Client uses ``handler``."""
    return AiosClient(
        base_url="http://test.invalid",
        api_key=api_key,
        transport=httpx.MockTransport(handler),
    )


def test_bearer_header_sent():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"ok": True})

    with _mock_client(handler) as client:
        result = client.request("GET", "/health")
    assert result == {"ok": True}
    assert captured["auth"] == "Bearer key-123"


def test_missing_api_key_sends_no_auth_header():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"ok": True})

    with _mock_client(handler, api_key=None) as client:
        client.request("GET", "/health")
    assert captured["auth"] is None


def test_204_returns_none():
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    with _mock_client(handler) as client:
        result = client.request("DELETE", "/v1/agents/x")
    assert result is None


def test_error_envelope_decoded():
    body = {"error": {"type": "not_found", "message": "agent not found", "detail": {"id": "x"}}}

    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json=body)

    with _mock_client(handler) as client, pytest.raises(AiosApiError) as excinfo:
        client.request("GET", "/v1/agents/x")
    err = excinfo.value
    assert err.status_code == 404
    assert err.error_type == "not_found"
    assert err.message == "agent not found"
    assert err.detail == {"id": "x"}


def test_non_envelope_error_body_handled_gracefully():
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    with _mock_client(handler) as client, pytest.raises(AiosApiError) as excinfo:
        client.request("GET", "/x")
    assert excinfo.value.status_code == 500
    assert "internal server error" in excinfo.value.message


def test_params_pruned_of_none():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["qs"] = (
            str(request.url.query, "utf-8")
            if isinstance(request.url.query, bytes)
            else request.url.query
        )
        return httpx.Response(200, json={"data": []})

    with _mock_client(handler) as client:
        client.request("GET", "/v1/agents", params={"limit": 5, "after": None, "kind": None})
    qs = seen["qs"]
    assert "limit=5" in qs
    assert "after" not in qs
    assert "kind" not in qs


def test_stream_session_parses_sse():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = (
            "event: event\n"
            'data: {"seq": 1}\n'
            "\n"
            "event: delta\n"
            'data: {"delta": "he"}\n'
            "\n"
            "event: done\n"
            "data: {}\n"
            "\n"
        )
        return httpx.Response(
            200,
            content=payload.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    with (
        _mock_client(handler) as client,
        client.stream_session("sess_1", after_seq=0) as messages,
    ):
        collected = list(messages)
    assert [m.event for m in collected] == ["event", "delta", "done"]
    assert json.loads(collected[0].data) == {"seq": 1}
    assert json.loads(collected[1].data) == {"delta": "he"}
