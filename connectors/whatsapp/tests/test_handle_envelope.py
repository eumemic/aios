"""Integration: daemon ``message`` notification → emit_inbound call shape."""

from __future__ import annotations

from pathlib import Path

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


async def test_handle_inbound_reads_attachment_bytes(
    connector: WhatsappConnector, tmp_path: Path
) -> None:
    # Daemon writes media to disk; connector reads bytes off-loop and
    # forwards (filename, bytes, content_type) tuples to emit_inbound.
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"jpeg-bytes")
    p = dm_payload(text="caption")
    p["attachments"] = [{"path": str(img), "mimetype": "image/jpeg", "filename": "photo.jpg"}]
    await connector._handle_inbound_message(CONNECTION_ID, p)

    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["content"] == "caption"
    assert kwargs["attachments"] == [("photo.jpg", b"jpeg-bytes", "image/jpeg")]


async def test_handle_inbound_attachment_only_emits_with_empty_text(
    connector: WhatsappConnector, tmp_path: Path
) -> None:
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"jpeg-bytes")
    p = dm_payload(text="")
    p["attachments"] = [{"path": str(img), "mimetype": "image/jpeg", "filename": "photo.jpg"}]
    await connector._handle_inbound_message(CONNECTION_ID, p)

    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["content"] == ""
    assert len(kwargs["attachments"]) == 1


async def test_handle_inbound_sticker_emoji_in_metadata(connector: WhatsappConnector) -> None:
    p = dm_payload(text="")
    p["sticker_emoji"] = "🎉"
    await connector._handle_inbound_message(CONNECTION_ID, p)

    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["metadata"]["sticker_emoji"] == "🎉"
    # Sticker-only messages carry empty content — the emoji IS the signal.
    assert kwargs["content"] == ""
    assert kwargs["attachments"] is None


async def test_handle_inbound_skips_unreadable_attachment(
    connector: WhatsappConnector, tmp_path: Path
) -> None:
    # If the daemon wrote two attachments and one got truncated /
    # removed before we read, surface the readable ones rather than
    # drop the whole message.
    good = tmp_path / "good.jpg"
    good.write_bytes(b"jpeg")
    bad = tmp_path / "missing.jpg"  # never created
    p = dm_payload(text="see attached")
    p["attachments"] = [
        {"path": str(bad), "mimetype": "image/jpeg", "filename": "bad.jpg"},
        {"path": str(good), "mimetype": "image/jpeg", "filename": "good.jpg"},
    ]
    await connector._handle_inbound_message(CONNECTION_ID, p)

    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["attachments"] == [("good.jpg", b"jpeg", "image/jpeg")]


async def test_handle_inbound_reaction_stamps_metadata(connector: WhatsappConnector) -> None:
    # Peer reacts to one of our messages — surface as
    # metadata.reaction with empty content; the model uses
    # target_message_id to match against its own send history.
    p = dm_payload(text="")
    p["reaction"] = {"emoji": "👍", "target_message_id": "3EB0OURMSG"}
    await connector._handle_inbound_message(CONNECTION_ID, p)

    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["content"] == ""
    assert kwargs["metadata"]["reaction"] == {
        "emoji": "👍",
        "target_message_id": "3EB0OURMSG",
    }


async def test_handle_inbound_reaction_removal_passes_through(
    connector: WhatsappConnector,
) -> None:
    p = dm_payload(text="")
    p["reaction"] = {"emoji": "", "target_message_id": "3EB0OURMSG"}
    await connector._handle_inbound_message(CONNECTION_ID, p)

    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["metadata"]["reaction"]["emoji"] == ""


async def test_handle_inbound_edit_stamps_metadata(connector: WhatsappConnector) -> None:
    # Peer edited their earlier message — content is the new body,
    # metadata.edited=True flags the rewrite for the harness's
    # context renderer.
    p = dm_payload(text="corrected text")
    p["edit"] = {"target_message_id": "3EB0ORIGINAL"}
    await connector._handle_inbound_message(CONNECTION_ID, p)

    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["content"] == "corrected text"
    assert kwargs["metadata"]["edited"] is True
    assert kwargs["metadata"]["edit_target_message_id"] == "3EB0ORIGINAL"


async def test_handle_inbound_revoke_stamps_metadata(connector: WhatsappConnector) -> None:
    p = dm_payload(text="")
    p["revoke"] = {"target_message_id": "3EB0ORIGINAL"}
    await connector._handle_inbound_message(CONNECTION_ID, p)

    kwargs = connector.emit_inbound.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["content"] == ""
    assert kwargs["metadata"]["revoked"] is True
    assert kwargs["metadata"]["revoke_target_message_id"] == "3EB0ORIGINAL"
