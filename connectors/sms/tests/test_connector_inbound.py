"""serve_connection inbound path: demux registration + queue drain →
emit_inbound, and the event_id single-source invariant (design §3.2)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from aios_sms.connector import (
    _EVENT_ID_FIELD,
    _FORBIDDEN_EVENT_ID_FIELDS,
    SmsConnector,
)
from aios_sms.webhook import InboundEnvelope

OUR_NUMBER = "+18005551234"
PEER = "+14155550000"


@pytest.fixture
def connector() -> SmsConnector:
    return SmsConnector()


async def test_emit_envelope_maps_twilio_fields(connector: SmsConnector) -> None:
    captured: list[dict[str, Any]] = []

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    env = InboundEnvelope(
        connection_id="conn_1",
        params={
            "From": PEER,
            "To": OUR_NUMBER,
            "Body": "hello world",
            "MessageSid": "SM_abc",
            "NumSegments": "2",
            # aliases present on the wire but MUST NOT be used as event_id
            "SmsSid": "SM_alias_should_be_ignored",
            "SmsMessageSid": "SM_alias_should_be_ignored",
        },
    )
    await connector._emit_envelope("conn_1", env)

    assert len(captured) == 1
    call = captured[0]
    assert call["connection_id"] == "conn_1"
    assert call["chat_id"] == PEER
    assert call["sender"] == {"display_name": PEER}
    assert call["content"] == "hello world"
    # event_id is the single-source MessageSid, never an alias.
    assert call["event_id"] == "SM_abc"
    # From provenance is stamped unverified toward the model.
    assert call["metadata"]["sender_verified"] is False
    assert call["metadata"]["num_segments"] == "2"


async def test_event_id_is_message_sid_not_aliases(connector: SmsConnector) -> None:
    """The single-source invariant: event_id derives from MessageSid and
    the SmsSid / SmsMessageSid aliases are never the source (design §3.2)."""
    assert _EVENT_ID_FIELD == "MessageSid"
    assert "SmsSid" in _FORBIDDEN_EVENT_ID_FIELDS
    assert "SmsMessageSid" in _FORBIDDEN_EVENT_ID_FIELDS

    captured: list[str | None] = []

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs.get("event_id"))
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    # Aliases differ from MessageSid; if the impl ever read an alias this
    # would surface as the wrong event_id.
    env = InboundEnvelope(
        connection_id="c",
        params={
            "From": PEER,
            "Body": "x",
            "MessageSid": "SM_real",
            "SmsSid": "SM_wrong",
            "SmsMessageSid": "SM_wrong",
        },
    )
    await connector._emit_envelope("c", env)
    assert captured == ["SM_real"]


async def test_emit_envelope_drops_when_no_message_sid(connector: SmsConnector) -> None:
    called = False

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]
    env = InboundEnvelope(connection_id="c", params={"From": PEER, "Body": "x"})
    await connector._emit_envelope("c", env)
    assert called is False


async def test_serve_connection_registers_demux_and_drains() -> None:
    connector = SmsConnector()
    emitted: list[dict[str, Any]] = []

    async def _fake_emit(**kwargs: Any) -> dict[str, Any]:
        emitted.append(kwargs)
        return {"deduped": False}

    connector.emit_inbound = _fake_emit  # type: ignore[method-assign]

    secrets = {"from_number": "1 800 555-1234", "auth_token": "tok"}
    task = asyncio.create_task(connector.serve_connection("conn_1", secrets))
    # Let serve_connection register before we route through the listener.
    await asyncio.sleep(0.05)

    # The demux map is keyed by the normalized number.
    entry = connector._listener.lookup(OUR_NUMBER)
    assert entry is not None
    assert entry.connection_id == "conn_1"

    # Push an envelope onto the registered queue; the drain loop emits it.
    entry.queue.put_nowait(
        InboundEnvelope(
            connection_id="conn_1",
            params={"From": PEER, "Body": "drained", "MessageSid": "SM_x"},
        )
    )
    await asyncio.sleep(0.05)
    assert emitted and emitted[0]["content"] == "drained"
    assert emitted[0]["event_id"] == "SM_x"

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    # On teardown the demux registration is removed.
    assert connector._listener.lookup(OUR_NUMBER) is None


async def test_serve_connection_requires_from_number_and_token() -> None:
    connector = SmsConnector()
    with pytest.raises(RuntimeError, match="from_number"):
        await connector.serve_connection("c", {"auth_token": "t"})
    with pytest.raises(RuntimeError, match="auth_token"):
        await connector.serve_connection("c", {"from_number": OUR_NUMBER})
