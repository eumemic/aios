"""Tests for parse_message — telegram.Message → InboundMessage —
and parse_reaction — MessageReactionUpdated → InboundReaction."""

from __future__ import annotations

from telegram import Message, MessageReactionUpdated

from aios_telegram.parse import parse_message, parse_reaction


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


def test_static_sticker_surfaces_webp_attachment(
    message_static_sticker: Message, bot_id: int
) -> None:
    msg = parse_message(message_static_sticker, bot_id=bot_id)
    assert msg is not None
    assert msg.text == ""
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.file_id == "STATIC-STICKER"
    assert att.content_type == "image/webp"
    assert att.filename.endswith(".webp")
    assert msg.sticker_emoji == "👍"


def test_video_sticker_surfaces_webm_attachment(
    message_video_sticker: Message, bot_id: int
) -> None:
    msg = parse_message(message_video_sticker, bot_id=bot_id)
    assert msg is not None
    assert msg.attachments[0].content_type == "video/webm"
    assert msg.attachments[0].filename.endswith(".webm")
    assert msg.sticker_emoji == "🎉"


def test_animation_surfaces_video_attachment(message_animation: Message, bot_id: int) -> None:
    msg = parse_message(message_animation, bot_id=bot_id)
    assert msg is not None
    # PTB also fills .document for animations; we surface the
    # animation slot which is more accurate.
    file_ids = {a.file_id for a in msg.attachments}
    assert "ANIM-A" in file_ids
    anim_atts = [a for a in msg.attachments if a.filename == "fun.mp4"]
    assert any(a.content_type == "video/mp4" for a in anim_atts)


def test_video_note_surfaces_video_attachment(message_video_note: Message, bot_id: int) -> None:
    msg = parse_message(message_video_note, bot_id=bot_id)
    assert msg is not None
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.file_id == "VN-A"
    assert att.content_type == "video/mp4"
    assert att.filename.startswith("video_note-")


def test_edited_message_marks_edited_flag(message_edited: Message, bot_id: int) -> None:
    msg = parse_message(message_edited, bot_id=bot_id)
    assert msg is not None
    assert msg.edited is True
    assert msg.text == "fixed typo"
    assert msg.message_id == 730


def test_reaction_added_emits_new_emoji(
    reaction_added: MessageReactionUpdated, bot_id: int
) -> None:
    parsed = parse_reaction(reaction_added, bot_id=bot_id)
    assert parsed is not None
    assert parsed.target_message_id == 901
    assert parsed.sender_id == 123456789
    assert parsed.new_emojis == ("👍",)
    assert parsed.old_emojis == ()


def test_reaction_removed_emits_old_emoji(
    reaction_removed: MessageReactionUpdated, bot_id: int
) -> None:
    parsed = parse_reaction(reaction_removed, bot_id=bot_id)
    assert parsed is not None
    assert parsed.new_emojis == ()
    assert parsed.old_emojis == ("👍",)


def test_reaction_anonymous_supergroup_returns_none(
    reaction_anonymous_in_supergroup: MessageReactionUpdated, bot_id: int
) -> None:
    # Anonymous reactions ride on actor_chat — punted in v1.
    assert parse_reaction(reaction_anonymous_in_supergroup, bot_id=bot_id) is None


def test_reaction_custom_emoji_only_returns_none(
    reaction_custom_emoji_only: MessageReactionUpdated, bot_id: int
) -> None:
    # Custom (premium) reactions are dropped — no glyph to surface.
    assert parse_reaction(reaction_custom_emoji_only, bot_id=bot_id) is None
