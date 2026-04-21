"""Tests for IngestClient retry behavior and metadata building."""

from __future__ import annotations

import httpx
import pytest

from aios_telegram import ingest as ingest_module
from aios_telegram.ingest import IngestClient, build_metadata
from aios_telegram.parse import InboundMessage, Reply


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # Drop backoff delays to 0 so tests run instantly.
    monkeypatch.setattr(ingest_module, "RETRY_DELAYS_SECONDS", (0.0, 0.0, 0.0, 0.0))


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected extra request")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


async def _run_post(transport: httpx.AsyncBaseTransport) -> None:
    async with IngestClient(
        base_url="http://aios.local", api_key="k", connection_id="conn_1"
    ) as client:
        await client._client.aclose()  # type: ignore[union-attr]
        client._client = httpx.AsyncClient(
            base_url="http://aios.local",
            headers={"Authorization": "Bearer k"},
            transport=transport,
        )
        await client.post_message(path="c1", content="hi", metadata={"a": 1})


async def test_success_first_try() -> None:
    transport = _RecordingTransport([httpx.Response(201, json={})])
    await _run_post(transport)
    assert len(transport.requests) == 1
    body = transport.requests[0].read()
    assert b'"path":"c1"' in body.replace(b" ", b"")
    assert b'"content":"hi"' in body.replace(b" ", b"")
    assert transport.requests[0].headers["authorization"] == "Bearer k"


async def test_client_error_no_retry() -> None:
    transport = _RecordingTransport([httpx.Response(422, json={"error": "bad"})])
    await _run_post(transport)
    assert len(transport.requests) == 1


async def test_server_error_retries_then_drops() -> None:
    # 5 attempts total (initial + 4 retries) all 503 → log and drop.
    transport = _RecordingTransport([httpx.Response(503, json={})] * 5)
    await _run_post(transport)
    assert len(transport.requests) == 5


async def test_network_error_retries_then_succeeds() -> None:
    transport = _RecordingTransport(
        [
            httpx.ConnectError("boom"),
            httpx.ConnectError("boom again"),
            httpx.Response(201, json={}),
        ]
    )
    await _run_post(transport)
    assert len(transport.requests) == 3


def test_build_metadata_dm_minimal() -> None:
    msg = InboundMessage(
        chat_kind="dm",
        chat_id=123,
        chat_name=None,
        sender_id=123,
        sender_name="Alice",
        message_id=1,
        timestamp_ms=1700000000000,
        text="hi",
        reply=None,
    )
    meta = build_metadata(msg, bot_id=999)
    assert meta == {
        "channel": "telegram/999/123",
        "chat_type": "dm",
        "sender_id": 123,
        "sender_name": "Alice",
        "message_id": 1,
        "timestamp_ms": 1700000000000,
    }


def test_build_metadata_group_with_reply() -> None:
    msg = InboundMessage(
        chat_kind="group",
        chat_id=-987,
        chat_name="Friends",
        sender_id=111,
        sender_name="Bob",
        message_id=2,
        timestamp_ms=1700000001000,
        text="hi",
        reply=Reply(message_id=1, text="earlier"),
    )
    meta = build_metadata(msg, bot_id=999)
    assert meta["channel"] == "telegram/999/-987"
    assert meta["chat_name"] == "Friends"
    assert meta["reply_to"] == {"message_id": 1, "text": "earlier"}
