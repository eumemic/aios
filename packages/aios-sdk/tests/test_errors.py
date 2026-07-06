"""Tests for the SDK's error-envelope decoder + raw-body request helper.

These moved wholesale from ``aios``'s ``tests/unit/cli/test_client.py`` when
the hand-written ``AiosClient`` collapsed onto the generated SDK client
(issue #1682): the envelope contract, param pruning, and transport-error
mapping are now SDK responsibilities, so they are tested here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from aios_sdk import (
    AiosApiError,
    Client,
    error_from_response,
    raw_request,
    stream_session,
)


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    # Route the transport through ``httpx_args`` (not ``set_httpx_client``) so
    # the SDK builds its own ``httpx.Client`` and still injects the Bearer
    # header — mirroring how the CLI's ``sdk_client`` factory works.
    return Client(
        base_url="http://test.invalid",
        token="key-123",
        httpx_args={"transport": httpx.MockTransport(handler)},
    )


def test_error_from_response_decodes_envelope() -> None:
    body = {"error": {"type": "not_found", "message": "agent not found", "detail": {"id": "x"}}}
    err = error_from_response(404, json.dumps(body))
    assert err.status_code == 404
    assert err.error_type == "not_found"
    assert err.message == "agent not found"
    assert err.detail == {"id": "x"}


def test_error_from_response_non_envelope_body_graceful() -> None:
    err = error_from_response(500, "internal server error")
    assert err.status_code == 500
    assert "internal server error" in err.message


def test_error_from_response_empty_body() -> None:
    err = error_from_response(502, b"")
    assert err.status_code == 502
    assert err.error_type == "http_error"


def test_raw_request_bearer_header_sent() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"ok": True})

    with _mock_client(handler) as client:
        result = raw_request(client, "GET", "/health")
    assert result == {"ok": True}
    assert captured["auth"] == "Bearer key-123"


def test_raw_request_204_returns_none() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    with _mock_client(handler) as client:
        assert raw_request(client, "DELETE", "/v1/agents/x") is None


def test_raw_request_raises_on_error_envelope() -> None:
    body = {"error": {"type": "not_found", "message": "agent not found", "detail": {"id": "x"}}}

    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json=body)

    with _mock_client(handler) as client, pytest.raises(AiosApiError) as excinfo:
        raw_request(client, "GET", "/v1/agents/x")
    assert excinfo.value.error_type == "not_found"
    assert excinfo.value.status_code == 404


def test_raw_request_prunes_none_params() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["qs"] = str(request.url.query, "utf-8")
        return httpx.Response(200, json={"data": []})

    with _mock_client(handler) as client:
        raw_request(
            client, "GET", "/v1/agents", params={"limit": 5, "after": None, "kind": None}
        )
    assert "limit=5" in seen["qs"]
    assert "after" not in seen["qs"]
    assert "kind" not in seen["qs"]


def test_raw_request_maps_connect_error() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with _mock_client(handler) as client, pytest.raises(AiosApiError) as excinfo:
        raw_request(client, "GET", "/health")
    assert excinfo.value.status_code == 0
    assert excinfo.value.error_type == "connection_error"


def test_stream_session_decodes_error_envelope() -> None:
    """A non-2xx on the SSE stream must decode the envelope, not leak httpx."""
    body = {"error": {"type": "not_found", "message": "no such session", "detail": {}}}

    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json=body)

    with _mock_client(handler) as client, pytest.raises(AiosApiError) as excinfo:  # noqa: SIM117
        with stream_session(client, "sess_x", after_seq=0) as messages:
            list(messages)
    assert excinfo.value.error_type == "not_found"
    assert excinfo.value.status_code == 404


def test_stream_session_parses_sse() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
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
        stream_session(client, "sess_1", after_seq=0) as messages,
    ):
        collected = list(messages)
    assert [m.event for m in collected] == ["event", "delta", "done"]
    assert json.loads(collected[0].data) == {"seq": 1}
