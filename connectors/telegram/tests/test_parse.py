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
    # PTB aliases ``animation`` into ``document`` for GIFs (same file_id +
    # file_name).  Surfacing both produced duplicate attachments that the
    # supervisor's staging layer refused — confirm we deduplicate to a
    # single video/mp4 attachment.
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.file_id == "ANIM-A"
    assert att.filename == "fun.mp4"
    assert att.content_type == "video/mp4"


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


# ── mentions ──────────────────────────────────────────────────────────


def test_no_mentions(message_dm_text: Message, bot_id: int) -> None:
    msg = parse_message(message_dm_text, bot_id=bot_id)
    assert msg is not None
    assert msg.mentions == ()


def test_text_mention_of_bot_populates_mentions(
    message_text_mention_of_bot: Message, bot_id: int
) -> None:
    msg = parse_message(message_text_mention_of_bot, bot_id=bot_id)
    assert msg is not None
    assert len(msg.mentions) == 1
    m = msg.mentions[0]
    assert m.user_id == bot_id
    assert m.name == "TestBot"


def test_text_mention_of_other_user_populates_mentions(
    message_text_mention_of_other_user: Message, bot_id: int
) -> None:
    msg = parse_message(message_text_mention_of_other_user, bot_id=bot_id)
    assert msg is not None
    assert len(msg.mentions) == 1
    m = msg.mentions[0]
    assert m.user_id == 444555666
    assert m.name == "Carol Doe"


def test_plain_username_mention_of_bot_with_bot_username_synthesizes(
    message_plain_username_mention_of_bot: Message, bot_id: int
) -> None:
    """``@<bot_username>`` carries no user_id on the wire, but we
    synthesize a structured mention when ``bot_username`` is supplied
    so the model gets a consistent signal for self-tagging."""
    msg = parse_message(
        message_plain_username_mention_of_bot, bot_id=bot_id, bot_username="testbot"
    )
    assert msg is not None
    assert len(msg.mentions) == 1
    m = msg.mentions[0]
    assert m.user_id == bot_id


def test_plain_username_mention_of_other_does_not_surface(
    message_plain_username_mention_of_other: Message, bot_id: int
) -> None:
    """Plain ``@user`` without text_mention carries no user_id; we don't
    invent one. Only ``text_mention`` and self-username matches surface."""
    msg = parse_message(
        message_plain_username_mention_of_other, bot_id=bot_id, bot_username="testbot"
    )
    assert msg is not None
    assert msg.mentions == ()


def test_plain_bot_mention_without_bot_username_does_not_synthesize(
    message_plain_username_mention_of_bot: Message, bot_id: int
) -> None:
    """When ``bot_username`` is not provided, plain ``@anything`` cannot
    be matched against the bot and produces no structured mention."""
    msg = parse_message(message_plain_username_mention_of_bot, bot_id=bot_id)
    assert msg is not None
    assert msg.mentions == ()


def test_mention_in_caption_parses_from_caption_entities(
    message_mention_in_caption: Message, bot_id: int
) -> None:
    msg = parse_message(message_mention_in_caption, bot_id=bot_id)
    assert msg is not None
    assert len(msg.mentions) == 1
    assert msg.mentions[0].user_id == bot_id


def test_plain_bot_mention_resolves_through_utf16_emoji_prefix(
    message_emoji_prefix_bot_mention: Message, bot_id: int
) -> None:
    """Regression guard: Telegram entity offsets are UTF-16 code units;
    a non-BMP char before the mention shifts the Python char index by
    the surrogate-pair count. Naive ``text[offset:offset+length]``
    misses the leading ``@`` and matches ``testbot `` instead of
    ``@testbot``, so the synthesis wouldn't fire."""
    msg = parse_message(message_emoji_prefix_bot_mention, bot_id=bot_id, bot_username="testbot")
    assert msg is not None
    assert len(msg.mentions) == 1
    assert msg.mentions[0].user_id == bot_id


def test_plain_bot_mention_in_caption_resolves_through_utf16(
    message_plain_bot_mention_in_caption: Message, bot_id: int
) -> None:
    """Caption_entities path: same UTF-16 invariant as text_entities,
    via ``message.parse_caption_entity``."""
    msg = parse_message(message_plain_bot_mention_in_caption, bot_id=bot_id, bot_username="testbot")
    assert msg is not None
    assert len(msg.mentions) == 1
    assert msg.mentions[0].user_id == bot_id


def test_synthesized_self_mention_uses_bot_display_name(
    message_plain_username_mention_of_bot: Message, bot_id: int
) -> None:
    """When ``bot_display_name`` is supplied, the synthesized self-mention
    surfaces the display name (matching ``text_mention``'s
    ``entity.user.full_name`` semantic) rather than the @-handle. This
    keeps the same user_id from carrying two different ``name`` values
    across events."""
    msg = parse_message(
        message_plain_username_mention_of_bot,
        bot_id=bot_id,
        bot_username="testbot",
        bot_display_name="TestBot",
    )
    assert msg is not None
    assert len(msg.mentions) == 1
    assert msg.mentions[0].name == "TestBot"


def test_multi_mention_preserves_order_and_self_detection(
    message_multi_mentions: Message, bot_id: int
) -> None:
    """A single message can carry multiple mentions across entity types.

    Order is preserved (text_mention of Carol → text_mention of bot →
    plain ``@testbot`` → all three surface), and the bot's identity
    appears twice without deduping — once via text_mention, once via
    plain-username synthesis."""
    msg = parse_message(
        message_multi_mentions,
        bot_id=bot_id,
        bot_username="testbot",
        bot_display_name="TestBot",
    )
    assert msg is not None
    assert [(m.user_id, m.name) for m in msg.mentions] == [
        (444555666, "Carol Doe"),
        (bot_id, "TestBot"),
        (bot_id, "TestBot"),
    ]
