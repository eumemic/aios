"""Integration: daemon ``message`` notification → emit_inbound call shape.

Tests dial the dispatch path through ``_handle_inbound_message``
directly with a wire-shaped params dict — the listener stream itself
is exercised in ``test_rpc.py`` and the loop wiring is exercised in
``test_daemon.py`` (against the fake daemon).
"""

from __future__ import annotations

from typing import Any

from aios_whatsapp.connector import WhatsappConnector

from .conftest import CONNECTION_ID, GROUP_JID, PEER_JID


def _dm_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "3EB0BB36C97D4F8C29A4",
        "timestamp_ms": 1700000000000,
        "from_jid": PEER_JID,
        "from_push_name": "Alice",
        "chat_jid": PEER_JID,
        "chat_type": "dm",
        "chat_name": None,
        "is_self": False,
        "text": "hello bot",
    }
    payload.update(overrides)
    return payload


async def test_handle_inbound_dm_emits_with_canonical_fields(
    connector: WhatsappConnector,
) -> None:
    await connector._handle_inbound_message(CONNECTION_ID, _dm_payload())

    connector.emit_inbound.assert_awaited_once()  # type: ignore[attr-defined]
    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["connection_id"] == CONNECTION_ID
    assert kwargs["event_id"] == f"whatsapp-{PEER_JID}-3EB0BB36C97D4F8C29A4"
    assert kwargs["chat_id"] == PEER_JID
    assert kwargs["sender"] == {"jid": PEER_JID, "display_name": "Alice"}
    assert kwargs["content"] == "hello bot"
    assert kwargs["timestamp"].startswith("2023-11-14")  # 1700000000000 ms is 2023-11-14 UTC
    md = kwargs["metadata"]
    assert md["chat_type"] == "dm"
    assert md["chat_jid"] == PEER_JID
    assert md["sender_jid"] == PEER_JID
    assert md["message_id"] == "3EB0BB36C97D4F8C29A4"
    assert md["timestamp_ms"] == 1700000000000


async def test_handle_inbound_group_carries_chat_name(connector: WhatsappConnector) -> None:
    await connector._handle_inbound_message(
        CONNECTION_ID,
        _dm_payload(chat_jid=GROUP_JID, chat_type="group", chat_name="Test Group"),
    )
    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["chat_id"] == GROUP_JID
    assert kwargs["metadata"]["chat_type"] == "group"
    assert kwargs["metadata"]["chat_name"] == "Test Group"


async def test_handle_inbound_drops_self_echo(connector: WhatsappConnector) -> None:
    await connector._handle_inbound_message(CONNECTION_ID, _dm_payload(is_self=True))
    connector.emit_inbound.assert_not_awaited()  # type: ignore[attr-defined]


async def test_handle_inbound_drops_empty_text(connector: WhatsappConnector) -> None:
    await connector._handle_inbound_message(CONNECTION_ID, _dm_payload(text=""))
    connector.emit_inbound.assert_not_awaited()  # type: ignore[attr-defined]


async def test_handle_inbound_drops_broadcast(connector: WhatsappConnector) -> None:
    await connector._handle_inbound_message(
        CONNECTION_ID,
        _dm_payload(chat_jid="12345@broadcast", chat_type="broadcast"),
    )
    connector.emit_inbound.assert_not_awaited()  # type: ignore[attr-defined]
