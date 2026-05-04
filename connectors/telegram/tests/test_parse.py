"""Tests for parse_message — telegram.Message → InboundMessage."""

from __future__ import annotations

from telegram import Message

from aios_telegram.parse import parse_message


def test_dm_text(message_dm_text: Message, bot_id: int) -> None:
    msg = parse_message(message_dm_text, bot_id=bot_id)
    assert msg is not None
    assert msg.chat_kind == "dm"
    assert msg.chat_id == 123456789
    assert msg.chat_name is None
    assert msg.sender_id == 123456789
    assert msg.sender_name == "Alice"
    assert msg.message_id == 101
    assert msg.text == "hello there"
    assert msg.timestamp_ms == 1700000000 * 1000
    assert msg.reply is None


def test_group_text(message_group_text: Message, bot_id: int) -> None:
    msg = parse_message(message_group_text, bot_id=bot_id)
    assert msg is not None
    assert msg.chat_kind == "group"
    assert msg.chat_id == -987654321
    assert msg.chat_name == "Friends"
    assert msg.sender_name == "Bob Smith"


def test_supergroup_text(message_supergroup_text: Message, bot_id: int) -> None:
    msg = parse_message(message_supergroup_text, bot_id=bot_id)
    assert msg is not None
    assert msg.chat_kind == "supergroup"
    assert msg.chat_id == -1001234567890
    assert msg.chat_name == "Big Chat"


def test_reply(message_reply: Message, bot_id: int) -> None:
    msg = parse_message(message_reply, bot_id=bot_id)
    assert msg is not None
    assert msg.text == "agreed!"
    assert msg.reply is not None
    assert msg.reply.message_id == 400
    assert msg.reply.text == "Original message"


def test_self_message_returns_none(message_from_self: Message, bot_id: int) -> None:
    assert parse_message(message_from_self, bot_id=bot_id) is None


def test_other_bot_returns_none(message_from_other_bot: Message, bot_id: int) -> None:
    assert parse_message(message_from_other_bot, bot_id=bot_id) is None


def test_photo_without_text_surfaces_attachment(
    message_photo_no_text: Message, bot_id: int
) -> None:
    msg = parse_message(message_photo_no_text, bot_id=bot_id)
    assert msg is not None
    assert msg.text == ""
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.content_type == "image/jpeg"
    assert att.filename.endswith(".jpg")


def test_photo_with_caption_surfaces_both(message_photo_with_caption: Message, bot_id: int) -> None:
    msg = parse_message(message_photo_with_caption, bot_id=bot_id)
    assert msg is not None
    assert msg.text == "look at this cat"
    assert len(msg.attachments) == 1
    # Largest PhotoSize wins.
    assert msg.attachments[0].file_id == "L1"


def test_voice_surfaces_attachment(message_voice: Message, bot_id: int) -> None:
    msg = parse_message(message_voice, bot_id=bot_id)
    assert msg is not None
    assert msg.text == ""
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.file_id == "VOICE-A"
    assert att.content_type == "audio/ogg"
    assert att.filename.endswith(".ogg")


def test_document_surfaces_attachment(message_document: Message, bot_id: int) -> None:
    msg = parse_message(message_document, bot_id=bot_id)
    assert msg is not None
    assert msg.text == ""
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.file_id == "DOC-A"
    assert att.content_type == "application/pdf"
    assert att.filename == "report.pdf"


def test_channel_post_without_sender_returns_none(
    message_channel_post_no_sender: Message, bot_id: int
) -> None:
    # Channel posts have no ``from_user`` — out of scope for v1.
    assert parse_message(message_channel_post_no_sender, bot_id=bot_id) is None
