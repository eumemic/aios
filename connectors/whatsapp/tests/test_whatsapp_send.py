"""Tests for the ``whatsapp_send`` tool method."""

from __future__ import annotations

from pathlib import Path

import pytest

from aios_whatsapp.connector import WhatsappConnector, _chat_type_from_jid

from .conftest import CONNECTION_ID, GROUP_JID, PEER_JID, PHONE


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
    # Result stamps the resolved focal channel + chat_type.  PEER_JID
    # ends with @s.whatsapp.net → dm.
    assert result == {
        "message_id": "3EB0NEW",
        "timestamp_ms": 1700000100000,
        "channel": f"whatsapp/{PHONE}/{PEER_JID}",
        "chat_type": "dm",
    }


def test_chat_type_from_jid() -> None:
    assert _chat_type_from_jid(PEER_JID) == "dm"
    assert _chat_type_from_jid(GROUP_JID) == "group"


async def test_whatsapp_send_group_passes_group_jid(connector: WhatsappConnector) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "3EB0GRP",
        "timestamp_ms": 1700000200000,
    }
    result = await connector.whatsapp_send(
        text="hello group",
        connection_id=CONNECTION_ID,
        chat_id=GROUP_JID,
    )
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with(  # type: ignore[attr-defined]
        "sendMessage",
        {"jid": GROUP_JID, "text": "hello group"},
    )
    # GROUP_JID ends with @g.us → group.
    assert result["channel"] == f"whatsapp/{PHONE}/{GROUP_JID}"
    assert result["chat_type"] == "group"


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
    assert result == {
        "message_id": "3EB0IMG",
        "timestamp_ms": 1700000200000,
        "channel": f"whatsapp/{PHONE}/{PEER_JID}",
        "chat_type": "dm",
    }


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


async def test_whatsapp_send_encodes_mentions_into_params(
    connector: WhatsappConnector,
) -> None:
    # Model writes @<E.164> in text; the connector pulls it out into
    # mentioned_jids (in JID form) and forwards both to the daemon.
    # The text-on-the-wire keeps the @<phone> literal because
    # WhatsApp clients render the mention pill from the JID list.
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "3EB0MENTION",
        "timestamp_ms": 1700000300000,
    }
    await connector.whatsapp_send(
        text="hey @+15551234567 about the thing",
        connection_id=CONNECTION_ID,
        chat_id=PEER_JID,
    )
    sent = connector.state[CONNECTION_ID].daemon.rpc.call.await_args.args[1]  # type: ignore[attr-defined]
    assert sent["mentioned_jids"] == ["15551234567@s.whatsapp.net"]
    # markdown_to_whatsapp is a no-op on this text; the @<phone>
    # literal survives intact.
    assert "@+15551234567" in sent["text"]


async def test_whatsapp_send_omits_mentions_key_when_no_mentions(
    connector: WhatsappConnector,
) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "3EB0PLAIN",
        "timestamp_ms": 1700000301000,
    }
    await connector.whatsapp_send(
        text="no tags here",
        connection_id=CONNECTION_ID,
        chat_id=PEER_JID,
    )
    sent = connector.state[CONNECTION_ID].daemon.rpc.call.await_args.args[1]  # type: ignore[attr-defined]
    assert "mentioned_jids" not in sent


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
