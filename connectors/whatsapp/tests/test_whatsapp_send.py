"""Tests for the ``whatsapp_send`` tool method."""

from __future__ import annotations

from pathlib import Path

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


async def test_whatsapp_send_forwards_attachments(
    connector: WhatsappConnector, tmp_path: Path
) -> None:
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # minimal JPEG SOI marker
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "3EB0IMG",
        "timestamp_ms": 1700000200000,
    }
    result = await connector.whatsapp_send(
        text="caption",
        attachments=[img],
        connection_id=CONNECTION_ID,
        chat_id=PEER_JID,
    )
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with(  # type: ignore[attr-defined]
        "sendMessage",
        {
            "jid": PEER_JID,
            "text": "caption",
            "attachments": [{"path": str(img), "mimetype": "image/jpeg", "filename": "photo.jpg"}],
        },
    )
    assert result == {"message_id": "3EB0IMG", "timestamp_ms": 1700000200000}


async def test_whatsapp_send_unknown_mimetype_falls_back_to_octet_stream(
    connector: WhatsappConnector, tmp_path: Path
) -> None:
    # mimetypes.guess_type returns (None, None) for unknown extensions.
    # The connector substitutes application/octet-stream so the daemon
    # has a non-empty value to classify on (catches as Document).
    blob = tmp_path / "mysterious.deadbeef"
    blob.write_bytes(b"random bytes")
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "3EB0DOC",
        "timestamp_ms": 1700000201000,
    }
    await connector.whatsapp_send(
        text="",
        attachments=[blob],
        connection_id=CONNECTION_ID,
        chat_id=PEER_JID,
    )
    call_args = connector.state[CONNECTION_ID].daemon.rpc.call.await_args  # type: ignore[attr-defined]
    sent_params = call_args.args[1]
    assert sent_params["attachments"][0]["mimetype"] == "application/octet-stream"
    assert sent_params["attachments"][0]["filename"] == "mysterious.deadbeef"


async def test_whatsapp_send_multi_attachment_preserves_order(
    connector: WhatsappConnector, tmp_path: Path
) -> None:
    # Multi-attachment ordering matters for the caption-on-first
    # semantic — the daemon takes the first one to carry the text.
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"%PDF-")
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "3EB0MULTI",
        "timestamp_ms": 1700000202000,
    }
    await connector.whatsapp_send(
        text="see attachments",
        attachments=[img, pdf],
        connection_id=CONNECTION_ID,
        chat_id=PEER_JID,
    )
    sent_params = connector.state[CONNECTION_ID].daemon.rpc.call.await_args.args[1]  # type: ignore[attr-defined]
    assert [a["filename"] for a in sent_params["attachments"]] == ["a.jpg", "b.pdf"]
    assert sent_params["attachments"][0]["mimetype"] == "image/jpeg"
    assert sent_params["attachments"][1]["mimetype"] == "application/pdf"
