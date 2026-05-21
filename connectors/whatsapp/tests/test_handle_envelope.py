"""Integration: daemon ``message`` notification → emit_inbound call shape."""

from __future__ import annotations

from aios_whatsapp.connector import WhatsappConnector

from .conftest import CONNECTION_ID, GROUP_JID, PEER_JID, dm_payload


async def test_handle_inbound_dm_emits_with_canonical_fields(
    connector: WhatsappConnector,
) -> None:
    await connector._handle_inbound_message(CONNECTION_ID, dm_payload())

    connector.emit_inbound.assert_awaited_once()  # type: ignore[attr-defined]
    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["connection_id"] == CONNECTION_ID
    assert kwargs["event_id"] == f"whatsapp-{PEER_JID}-3EB0BB36C97D4F8C29A4"
    assert kwargs["chat_id"] == PEER_JID
    assert kwargs["sender"] == {"jid": PEER_JID, "display_name": "Alice"}
    assert kwargs["content"] == "hello bot"
    # 1700000000000 ms is 2023-11-14 UTC; iso_from_ms formats with offset.
    assert kwargs["timestamp"].startswith("2023-11-14")
    assert kwargs["metadata"] == {
        "chat_type": "dm",
        "sender_jid": PEER_JID,
        "message_id": "3EB0BB36C97D4F8C29A4",
    }


async def test_handle_inbound_group_carries_chat_name(connector: WhatsappConnector) -> None:
    await connector._handle_inbound_message(
        CONNECTION_ID,
        dm_payload(chat_jid=GROUP_JID, chat_type="group", chat_name="Test Group"),
    )
    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["chat_id"] == GROUP_JID
    assert kwargs["metadata"]["chat_type"] == "group"
    assert kwargs["metadata"]["chat_name"] == "Test Group"


async def test_handle_inbound_omits_chat_name_when_absent(connector: WhatsappConnector) -> None:
    await connector._handle_inbound_message(CONNECTION_ID, dm_payload())
    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert "chat_name" not in kwargs["metadata"]


async def test_handle_inbound_drops_self_echo(connector: WhatsappConnector) -> None:
    await connector._handle_inbound_message(CONNECTION_ID, dm_payload(is_self=True))
    connector.emit_inbound.assert_not_awaited()  # type: ignore[attr-defined]


async def test_handle_inbound_drops_empty_text(connector: WhatsappConnector) -> None:
    await connector._handle_inbound_message(CONNECTION_ID, dm_payload(text=""))
    connector.emit_inbound.assert_not_awaited()  # type: ignore[attr-defined]


async def test_handle_inbound_drops_broadcast(connector: WhatsappConnector) -> None:
    await connector._handle_inbound_message(
        CONNECTION_ID,
        dm_payload(chat_jid="12345@broadcast", chat_type="broadcast"),
    )
    connector.emit_inbound.assert_not_awaited()  # type: ignore[attr-defined]
