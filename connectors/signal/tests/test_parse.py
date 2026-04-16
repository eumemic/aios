"""Tests for parse_envelope — signal-cli JSON → InboundMessage."""

from __future__ import annotations

from typing import Any

from aios_signal.parse import build_content_text, parse_envelope


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


def test_build_content_text_with_attachments(
    envelope_attachment_only: dict[str, Any], bot_uuid: str
) -> None:
    msg = parse_envelope(envelope_attachment_only, bot_account_uuid=bot_uuid)
    assert msg is not None
    rendered = build_content_text(msg)
    assert rendered == "[attachment: photo.jpg (image/jpeg)]"


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
        attachments=(Attachment(content_type="image/png", filename="x.png", signal_file=None),),
        reply=None,
        reaction=None,
    )
    rendered = build_content_text(msg)
    assert rendered == "Check this out\n[attachment: x.png (image/png)]"
