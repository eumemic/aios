from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from mautrix.types import EventID, EventType, RoomID, UserID

from aios_matrix.connector import MatrixConnector, ReceiverAction


@pytest.fixture
def connector(tmp_path):
    item = MatrixConnector(
        base_url="http://aios", token="token", spool_path=tmp_path / "answered.sqlite"
    )
    item.config = MagicMock(
        server_name="example.org", user_namespace_regex=r"^@_aios_agent_[a-z]+:example\.org$"
    )
    item.az = MagicMock()
    return item


async def test_matrix_send_uses_tool_call_id_as_matrix_txn(connector):
    intent = connector.az.intent.user.return_value
    intent.send_message_event = AsyncMock(return_value=EventID("$sent"))

    result = await connector.matrix_send(
        connection_id="con",
        external_account_id="_aios_agent_one",
        chat_id="!dm:example.org",
        text="hello",
        format="plain",
        tool_call_id="call_123",
    )

    assert result == {"event_id": "$sent", "room_id": "!dm:example.org"}
    assert intent.send_message_event.await_args.kwargs["txn_id"] == "call_123"
    assert intent.send_message_event.await_args.args[:2] == (
        RoomID("!dm:example.org"),
        EventType.ROOM_MESSAGE,
    )


async def test_inbound_nonfatal_drop_continues_to_sibling(connector):
    connector._ghost_connections = {"_aios_agent_one": "c1", "_aios_agent_two": "c2"}
    # The membership view must be declared COMPLETE, or the connector
    # (correctly) refuses to classify on it rather than routing off a
    # possibly-partial member list.
    connector.az.state_store.has_full_member_list = AsyncMock(return_value=True)
    connector.az.state_store.get_members = AsyncMock(
        return_value=[
            UserID("@_aios_agent_one:example.org"),
            UserID("@_aios_agent_two:example.org"),
        ]
    )
    connector.emit_inbound = AsyncMock(side_effect=[None, {"deduped": False}])
    event = MagicMock(
        type=EventType.ROOM_MESSAGE,
        room_id=RoomID("!dm:example.org"),
        event_id=EventID("$evt"),
        sender=UserID("@human:example.org"),
        content=MagicMock(body="hi"),
        timestamp=1,
    )

    await connector._handle_event(event)

    assert connector.emit_inbound.await_count == 2


async def test_receiver_failure_table(connector):
    request = httpx.Request("POST", "http://aios")
    assert (
        connector._classify_receiver_failure(httpx.ConnectError("down", request=request))
        is ReceiverAction.RETRY
    )
    assert (
        connector._classify_receiver_failure(
            httpx.HTTPStatusError(
                "bad", request=request, response=httpx.Response(503, request=request)
            )
        )
        is ReceiverAction.RETRY
    )
    assert (
        connector._classify_receiver_failure(
            httpx.HTTPStatusError(
                "bad", request=request, response=httpx.Response(401, request=request)
            )
        )
        is ReceiverAction.HALT
    )
    assert connector._classify_receiver_failure(ValueError("poison")) is ReceiverAction.HALT


async def test_halt_signals_supervisor(connector):
    connector._halt = asyncio.Event()
    await connector._receiver_failed(ValueError("poison"))
    assert connector._halt.is_set()
