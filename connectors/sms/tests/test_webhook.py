"""Webhook listener — verify-before-parse, route-then-verify, cold-start.

These drive the aiohttp app via ``aiohttp.test_utils`` (no real socket,
no network) and assert the security properties of the inbound/status
routes (design §3.2, §5.1, §5.3).
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from aios_sms.config import EMPTY_TWIML
from aios_sms.verify import compute_signature
from aios_sms.webhook import (
    TWILIO_SIGNATURE_HEADER,
    DemuxEntry,
    InboundQueue,
    WebhookListener,
)

BASE = "https://sms.example.com"
OUR_NUMBER = "+18005551234"
PEER = "+14155550000"
AUTH_TOKEN = "test-auth-token"


def _make_listener() -> tuple[WebhookListener, InboundQueue]:
    listener = WebhookListener(public_base_url=BASE)
    queue: InboundQueue = InboundQueue(maxsize=10)
    listener.register(
        OUR_NUMBER,
        DemuxEntry(connection_id="conn_1", auth_token=AUTH_TOKEN, queue=queue),
    )
    return listener, queue


def _sign(path: str, params: dict[str, str]) -> str:
    return compute_signature(AUTH_TOKEN, BASE + path, params)


@pytest.fixture
async def client() -> TestClient:
    listener, queue = _make_listener()
    server = TestServer(listener.app)
    c = TestClient(server)
    await c.start_server()
    c.queue = queue  # type: ignore[attr-defined]
    c.listener = listener  # type: ignore[attr-defined]
    yield c
    await c.close()


async def test_signed_inbound_verifies_and_enqueues(client: TestClient) -> None:
    params = {"To": OUR_NUMBER, "From": PEER, "Body": "hello", "MessageSid": "SM123"}
    resp = await client.post(
        "/twilio/inbound",
        data=params,
        headers={TWILIO_SIGNATURE_HEADER: _sign("/twilio/inbound", params)},
    )
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("text/xml")
    assert (await resp.text()) == EMPTY_TWIML
    # Enqueued for the drain loop, off the webhook critical path.
    env = client.queue.get_nowait()  # type: ignore[attr-defined]
    assert env.connection_id == "conn_1"
    assert env.params["MessageSid"] == "SM123"
    assert env.params["Body"] == "hello"


async def test_unsigned_inbound_is_403(client: TestClient) -> None:
    params = {"To": OUR_NUMBER, "From": PEER, "Body": "hello", "MessageSid": "SM1"}
    resp = await client.post("/twilio/inbound", data=params)  # no signature header
    assert resp.status == 403
    assert client.queue.empty()  # type: ignore[attr-defined]


async def test_forged_signature_is_403(client: TestClient) -> None:
    params = {"To": OUR_NUMBER, "From": PEER, "Body": "hello", "MessageSid": "SM1"}
    resp = await client.post(
        "/twilio/inbound",
        data=params,
        headers={TWILIO_SIGNATURE_HEADER: "not-a-valid-signature"},
    )
    assert resp.status == 403
    assert client.queue.empty()  # type: ignore[attr-defined]


async def test_tampered_body_is_403(client: TestClient) -> None:
    """Sign one body, send a different one — fail closed."""
    signed = {"To": OUR_NUMBER, "From": PEER, "Body": "hello", "MessageSid": "SM1"}
    sig = _sign("/twilio/inbound", signed)
    tampered = dict(signed, Body="malicious")
    resp = await client.post(
        "/twilio/inbound",
        data=tampered,
        headers={TWILIO_SIGNATURE_HEADER: sig},
    )
    assert resp.status == 403
    assert client.queue.empty()  # type: ignore[attr-defined]


async def test_cold_start_returns_transient_5xx_not_403(client: TestClient) -> None:
    """A webhook for an undiscovered number → transient 5xx so Twilio
    retries; never 403 (briefly unavailable, never unauthenticated)."""
    unknown = "+19998887777"
    params = {"To": unknown, "From": PEER, "Body": "hi", "MessageSid": "SM1"}
    # Even with a *valid-looking* signature for some token, the number
    # isn't routable yet, so we 5xx before any verify.
    resp = await client.post(
        "/twilio/inbound",
        data=params,
        headers={TWILIO_SIGNATURE_HEADER: "anything"},
    )
    assert resp.status >= 500
    assert resp.status != 403
    assert client.queue.empty()  # type: ignore[attr-defined]


async def test_route_then_verify_uses_correct_connection_token(client: TestClient) -> None:
    """Two connections with distinct tokens: a request signed with conn A's
    token but addressed (To) to conn B's number must fail closed against
    B's token — never a silent cross-connection accept (design §3.2)."""
    other_queue: InboundQueue = InboundQueue(maxsize=10)
    other_number = "+18005559999"
    client.listener.register(  # type: ignore[attr-defined]
        other_number,
        DemuxEntry(connection_id="conn_2", auth_token="conn-2-token", queue=other_queue),
    )
    # Sign with conn_1's token (AUTH_TOKEN) but address To=conn_2's number.
    params = {"To": other_number, "From": PEER, "Body": "x", "MessageSid": "SM1"}
    sig = compute_signature(AUTH_TOKEN, BASE + "/twilio/inbound", params)
    resp = await client.post(
        "/twilio/inbound",
        data=params,
        headers={TWILIO_SIGNATURE_HEADER: sig},
    )
    assert resp.status == 403
    assert other_queue.empty()
    assert client.queue.empty()  # type: ignore[attr-defined]


async def test_status_callback_routes_by_from_and_verifies(client: TestClient) -> None:
    """Status callbacks route by the From number (our owned number on an
    outbound message) — the opposite axis from inbound (design §3.5)."""
    params = {
        "From": OUR_NUMBER,
        "To": PEER,
        "MessageSid": "SM9",
        "MessageStatus": "delivered",
    }
    resp = await client.post(
        "/twilio/status",
        data=params,
        headers={TWILIO_SIGNATURE_HEADER: _sign("/twilio/status", params)},
    )
    assert resp.status == 204


async def test_status_callback_unsigned_is_403(client: TestClient) -> None:
    params = {"From": OUR_NUMBER, "To": PEER, "MessageSid": "SM9", "MessageStatus": "failed"}
    resp = await client.post("/twilio/status", data=params)
    assert resp.status == 403


async def test_queue_overflow_sheds_but_still_acks_200() -> None:
    """A full per-connection queue sheds the inbound but still acks 200 —
    a dropped inbound is recoverable via Twilio retry; an OOM is not."""
    listener = WebhookListener(public_base_url=BASE)
    queue: InboundQueue = InboundQueue(maxsize=1)
    queue.put_nowait(  # pre-fill so the next put overflows
        __import__("aios_sms.webhook", fromlist=["InboundEnvelope"]).InboundEnvelope(
            connection_id="conn_1", params={}
        )
    )
    listener.register(
        OUR_NUMBER, DemuxEntry(connection_id="conn_1", auth_token=AUTH_TOKEN, queue=queue)
    )
    server = TestServer(listener.app)
    c = TestClient(server)
    await c.start_server()
    try:
        params = {"To": OUR_NUMBER, "From": PEER, "Body": "x", "MessageSid": "SM2"}
        resp = await c.post(
            "/twilio/inbound",
            data=params,
            headers={TWILIO_SIGNATURE_HEADER: _sign("/twilio/inbound", params)},
        )
        assert resp.status == 200
        assert queue.qsize() == 1  # shed, not grown
    finally:
        await c.close()


async def test_offloop_verify_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default verify path runs off the event loop via to_thread."""
    calls: list[str] = []
    real_to_thread = asyncio.to_thread

    async def _tracking(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("to_thread")
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _tracking)
    listener, _q = _make_listener()
    server = TestServer(listener.app)
    c = TestClient(server)
    await c.start_server()
    try:
        params = {"To": OUR_NUMBER, "From": PEER, "Body": "x", "MessageSid": "SM3"}
        await c.post(
            "/twilio/inbound",
            data=params,
            headers={TWILIO_SIGNATURE_HEADER: _sign("/twilio/inbound", params)},
        )
        assert "to_thread" in calls
    finally:
        await c.close()
