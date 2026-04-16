"""Tests for IngestClient retry behavior."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from aios_signal import ingest as ingest_module
from aios_signal.ingest import IngestClient, build_metadata
from aios_signal.parse import Attachment, InboundMessage, Reaction, Reply


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
        # Replace the underlying AsyncClient with one using our mock transport.
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
    # No retry on 4xx — exactly one request.
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


def test_build_metadata_minimal() -> None:
    msg = InboundMessage(
        chat_type="dm",
        raw_chat_id="u",
        sender_uuid="u",
        sender_name=None,
        chat_name=None,
        timestamp_ms=100,
        text="hi",
        attachments=(),
        reply=None,
        reaction=None,
    )
    md = build_metadata(msg, chat_id="u", bot_uuid="b")
    assert md == {
        "channel": "signal/b/u",
        "sender_uuid": "u",
        "timestamp_ms": 100,
        "chat_type": "dm",
    }


def test_build_metadata_full() -> None:
    msg = InboundMessage(
        chat_type="group",
        raw_chat_id="gid",
        sender_uuid="alice-uuid",
        sender_name="Alice",
        chat_name="Friends",
        timestamp_ms=100,
        text="hi",
        attachments=(Attachment(content_type="image/png", filename="x.png", signal_file=None),),
        reply=Reply(author_uuid="b", timestamp_ms=99, text="prev"),
        reaction=Reaction(emoji="👍", target_author_uuid="c", target_timestamp_ms=98),
    )
    md = build_metadata(msg, chat_id="gid-urlsafe", bot_uuid="bot")
    assert md["channel"] == "signal/bot/gid-urlsafe"
    assert md["chat_type"] == "group"
    assert md["sender_name"] == "Alice"
    assert md["chat_name"] == "Friends"
    assert md["reply_to"] == {"author_uuid": "b", "timestamp_ms": 99, "text": "prev"}
    assert md["reaction"] == {
        "emoji": "👍",
        "target_author_uuid": "c",
        "target_timestamp_ms": 98,
    }


async def test_post_message_outside_ctx_raises() -> None:
    client = IngestClient(base_url="x", api_key="k", connection_id="c")
    with pytest.raises(RuntimeError):
        await client.post_message(path="p", content="c", metadata={})


async def test_build_metadata_does_not_mutate_inputs() -> None:
    msg = InboundMessage(
        chat_type="dm",
        raw_chat_id="u",
        sender_uuid="u",
        sender_name=None,
        chat_name=None,
        timestamp_ms=1,
        text="hi",
        attachments=(),
        reply=None,
        reaction=None,
    )
    md1 = build_metadata(msg, chat_id="u", bot_uuid="b")
    md2 = build_metadata(msg, chat_id="u", bot_uuid="b")
    md1["extra"] = "mutation"
    assert "extra" not in md2


# Silence unused-type-import pragmas
_: Any = None
