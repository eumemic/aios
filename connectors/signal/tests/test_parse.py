"""Tests for parse_envelope — signal-cli JSON → InboundMessage."""

from __future__ import annotations

from typing import Any

from aios_signal.parse import build_content_text, is_group_update_envelope, parse_envelope


def test_text_dm(envelope_text_dm: dict[str, Any], bot_uuid: str) -> None:
    msg = parse_envelope(envelope_text_dm, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert msg.chat_type == "dm"
    assert msg.raw_chat_id == "11111111-2222-3333-4444-555555555555"
    assert msg.sender_uuid == "11111111-2222-3333-4444-555555555555"
    assert msg.sender_name == "Alice"
    assert msg.chat_name is None
    assert msg.text == "Hello there"
    assert msg.timestamp_ms == 1700000000000
    assert msg.attachments == ()
    assert msg.reply is None
    assert msg.reaction is None


def test_text_group_with_mentions(envelope_text_group: dict[str, Any], bot_uuid: str) -> None:
    msg = parse_envelope(envelope_text_group, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert msg.chat_type == "group"
    # raw (standard) base64, not URL-safe — addressing.py handles that.
    assert msg.raw_chat_id == "abc+def/xyz=="
    assert msg.chat_name == "Friends"
    # Mention placeholder substituted with @Name.
    assert msg.text == "hey @Bob thanks!"
    # Structured mention list preserved alongside the substituted text so
    # callers can distinguish a structured @-mention from a literal "@Bob".
    assert len(msg.mentions) == 1
    assert msg.mentions[0].uuid == "66666666-7777-8888-9999-aaaaaaaaaaaa"
    assert msg.mentions[0].name == "Bob"


def test_no_mentions_yields_empty_tuple(envelope_text_dm: dict[str, Any], bot_uuid: str) -> None:
    msg = parse_envelope(envelope_text_dm, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert msg.mentions == ()


def test_edit_envelope_surfaces_as_inbound(bot_uuid: str) -> None:
    """signal-cli wraps inbound peer edits as ``envelope.editMessage.dataMessage``
    rather than a top-level ``dataMessage``.  Without parsing both shapes the
    bot never sees that a peer rewrote a prior message; the envelope drops to
    None and the conversation thread silently diverges from the chat client's view.
    """
    envelope: dict[str, Any] = {
        "sourceUuid": "11111111-2222-3333-4444-555555555555",
        "sourceName": "Alice",
        "timestamp": 1700000010000,
        "editMessage": {
            "targetSentTimestamp": 1700000005000,
            "dataMessage": {
                "timestamp": 1700000010000,
                "message": "Actually, I meant Tuesday",
                "groupInfo": {
                    "groupId": "abc+def/xyz==",
                    "groupName": "Friends",
                    "type": "DELIVER",
                },
            },
        },
    }
    msg = parse_envelope(envelope, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert msg.text == "Actually, I meant Tuesday"
    assert msg.edited is True
    assert msg.edit_target_timestamp_ms == 1700000005000
    # New edit timestamp at the envelope root becomes the event's
    # ``timestamp_ms`` — uniquely identifies the edit relative to the original.
    assert msg.timestamp_ms == 1700000010000


def test_reaction(envelope_reaction: dict[str, Any], bot_uuid: str) -> None:
    msg = parse_envelope(envelope_reaction, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert msg.reaction is not None
    assert msg.reaction.emoji == "\U0001f44d"
    assert msg.reaction.target_author_uuid == "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    assert msg.reaction.target_timestamp_ms == 1699999999000
    assert msg.text == ""


def test_reply_has_quote(envelope_reply: dict[str, Any], bot_uuid: str) -> None:
    msg = parse_envelope(envelope_reply, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert msg.text == "agreed!"
    assert msg.reply is not None
    assert msg.reply.author_uuid == "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    assert msg.reply.timestamp_ms == 1699999000000
    assert msg.reply.text == "Original message"


def test_attachment_only(envelope_attachment_only: dict[str, Any], bot_uuid: str) -> None:
    msg = parse_envelope(envelope_attachment_only, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert msg.text == ""
    assert len(msg.attachments) == 1
    assert msg.attachments[0].content_type == "image/jpeg"
    assert msg.attachments[0].filename == "photo.jpg"


def test_self_message_returns_none(envelope_self: dict[str, Any], bot_uuid: str) -> None:
    assert parse_envelope(envelope_self, bot_account_uuid=bot_uuid) is None


def test_receipt_returns_none(envelope_receipt: dict[str, Any], bot_uuid: str) -> None:
    assert parse_envelope(envelope_receipt, bot_account_uuid=bot_uuid) is None


def test_typing_returns_none(envelope_typing: dict[str, Any], bot_uuid: str) -> None:
    assert parse_envelope(envelope_typing, bot_account_uuid=bot_uuid) is None


def test_missing_source_uuid_returns_none(bot_uuid: str) -> None:
    envelope: dict[str, Any] = {
        "timestamp": 1,
        "dataMessage": {"message": "hi", "timestamp": 1},
    }
    assert parse_envelope(envelope, bot_account_uuid=bot_uuid) is None


def test_build_content_text_attachment_only_is_empty(
    envelope_attachment_only: dict[str, Any], bot_uuid: str
) -> None:
    msg = parse_envelope(envelope_attachment_only, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert build_content_text(msg) == ""


def test_build_content_text_with_text_and_attachment(bot_uuid: str) -> None:
    from aios_signal.parse import Attachment, InboundMessage

    msg = InboundMessage(
        chat_type="dm",
        raw_chat_id="u",
        sender_uuid="u",
        sender_name=None,
        chat_name=None,
        timestamp_ms=1,
        text="Check this out",
        attachments=(
            Attachment(
                content_type="image/png",
                filename="x.png",
                host_path="/tmp/x",
                id="abc",
            ),
        ),
        mentions=(),
        reply=None,
        reaction=None,
    )
    assert build_content_text(msg) == "Check this out"


def test_attachment_captures_host_path(
    envelope_attachment_only: dict[str, Any], bot_uuid: str
) -> None:
    msg = parse_envelope(envelope_attachment_only, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert msg.attachments[0].host_path == "/tmp/signal-cli/attachments/abc-def-123"
    assert msg.attachments[0].id == "abc-def-123"


def test_attachment_no_file_field_carries_id(
    envelope_attachment_no_file_field: dict[str, Any], bot_uuid: str
) -> None:
    """JSON-RPC daemon mode envelopes omit ``file`` but keep ``id``;
    connector-side fallback uses ``id`` to reconstruct the on-disk path."""
    msg = parse_envelope(envelope_attachment_no_file_field, bot_account_uuid=bot_uuid)
    assert msg is not None
    assert msg.attachments[0].host_path is None
    assert msg.attachments[0].id == "xyz-789"


# ── is_group_update_envelope ──────────────────────────────────────────


def test_is_group_update_envelope_detects_update() -> None:
    envelope = {"dataMessage": {"groupInfo": {"groupId": "abc==", "type": "UPDATE"}}}
    assert is_group_update_envelope(envelope) is True


def test_is_group_update_envelope_false_for_deliver(
    envelope_text_group: dict[str, Any],
) -> None:
    assert is_group_update_envelope(envelope_text_group) is False


def test_is_group_update_envelope_false_for_dm(
    envelope_text_dm: dict[str, Any],
) -> None:
    assert is_group_update_envelope(envelope_text_dm) is False


def test_is_group_update_envelope_false_for_no_data_message() -> None:
    assert is_group_update_envelope({"sourceUuid": "xxx"}) is False


def test_is_group_update_envelope_false_for_no_group_info() -> None:
    assert is_group_update_envelope({"dataMessage": {"message": "hi"}}) is False
