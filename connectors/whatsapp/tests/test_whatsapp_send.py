"""Tests for the ``whatsapp_send`` tool method."""

from __future__ import annotations

import pytest

from aios_whatsapp.connector import WhatsappConnector

from .conftest import CONNECTION_ID, GROUP_JID, PEER_JID


async def test_whatsapp_send_dm_calls_send_message_rpc(connector: WhatsappConnector) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "3EB0NEW",
        "timestamp_ms": 1700000100000,
    }
    result = await connector.whatsapp_send(
        text="hello peer",
        connection_id=CONNECTION_ID,
        chat_id=PEER_JID,
    )
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with(  # type: ignore[attr-defined]
        "sendMessage",
        {"jid": PEER_JID, "text": "hello peer"},
    )
    assert result == {"message_id": "3EB0NEW", "timestamp_ms": 1700000100000}


async def test_whatsapp_send_group_passes_group_jid(connector: WhatsappConnector) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "3EB0GRP",
        "timestamp_ms": 1700000200000,
    }
    await connector.whatsapp_send(
        text="hello group",
        connection_id=CONNECTION_ID,
        chat_id=GROUP_JID,
    )
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with(  # type: ignore[attr-defined]
        "sendMessage",
        {"jid": GROUP_JID, "text": "hello group"},
    )


async def test_whatsapp_send_rejects_invalid_chat_id(connector: WhatsappConnector) -> None:
    with pytest.raises(ValueError, match="invalid WhatsApp chat_id"):
        await connector.whatsapp_send(
            text="hi",
            connection_id=CONNECTION_ID,
            chat_id="not-a-jid",
        )
    # The daemon must not be touched on a validation failure.
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_not_awaited()  # type: ignore[attr-defined]


async def test_whatsapp_send_raises_on_non_dict_result(connector: WhatsappConnector) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = "not-a-dict"  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="sendMessage returned non-dict"):
        await connector.whatsapp_send(
            text="hi",
            connection_id=CONNECTION_ID,
            chat_id=PEER_JID,
        )


async def test_whatsapp_send_raises_on_unknown_connection(connector: WhatsappConnector) -> None:
    with pytest.raises(KeyError):
        await connector.whatsapp_send(
            text="hi",
            connection_id="conn_unknown",
            chat_id=PEER_JID,
        )
