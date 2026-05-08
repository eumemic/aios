"""Shared test fixtures.

PTB's ``Message.de_json`` needs a ``Bot`` (or close enough) for its date
deserialization to resolve the right tzinfo. We pass a ``MagicMock(spec=Bot)``
with ``defaults=None`` — that's the minimum PTB will accept without opening
a network connection.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from telegram import Bot, Message

# HttpConnector reads AIOS_URL / AIOS_CONNECTOR_TOKEN at __init__ time, so
# every test that constructs a TelegramConnector needs them in env.  Set
# at import time rather than in a fixture so module-level connector
# instances also see them.
os.environ.setdefault("AIOS_URL", "http://test")
os.environ.setdefault("AIOS_CONNECTOR_TOKEN", "aios_conn_test")

BOT_ID = 99999999


@pytest.fixture
def bot_id() -> int:
    return BOT_ID


@pytest.fixture
def ptb_bot() -> Bot:
    """Minimal Bot double for ``Message.de_json`` construction."""
    bot = MagicMock(spec=Bot)
    bot.defaults = None
    return bot


def _make_message(data: dict[str, Any], bot: Bot) -> Message:
    msg = Message.de_json(data, bot)
    assert msg is not None
    return msg


@pytest.fixture
def message_dm_text(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 101,
            "date": 1700000000,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "text": "hello there",
        },
        ptb_bot,
    )


@pytest.fixture
def message_group_text(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 202,
            "date": 1700000001,
            "chat": {"id": -987654321, "type": "group", "title": "Friends"},
            "from": {
                "id": 111222333,
                "is_bot": False,
                "first_name": "Bob",
                "last_name": "Smith",
            },
            "text": "hey everyone",
        },
        ptb_bot,
    )


@pytest.fixture
def message_supergroup_text(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 303,
            "date": 1700000002,
            "chat": {"id": -1001234567890, "type": "supergroup", "title": "Big Chat"},
            "from": {"id": 444555666, "is_bot": False, "first_name": "Carol"},
            "text": "in the supergroup",
        },
        ptb_bot,
    )


@pytest.fixture
def message_reply(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 404,
            "date": 1700000003,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "text": "agreed!",
            "reply_to_message": {
                "message_id": 400,
                "date": 1699999999,
                "chat": {"id": 123456789, "type": "private"},
                "from": {"id": BOT_ID, "is_bot": True, "first_name": "Bot"},
                "text": "Original message",
            },
        },
        ptb_bot,
    )


@pytest.fixture
def message_from_self(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 505,
            "date": 1700000004,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": BOT_ID, "is_bot": True, "first_name": "Bot"},
            "text": "I'm the bot",
        },
        ptb_bot,
    )


@pytest.fixture
def message_from_other_bot(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 606,
            "date": 1700000005,
            "chat": {"id": -987654321, "type": "group", "title": "Friends"},
            "from": {"id": 888888888, "is_bot": True, "first_name": "OtherBot"},
            "text": "bot-to-bot",
        },
        ptb_bot,
    )


@pytest.fixture
def message_photo_no_text(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 707,
            "date": 1700000006,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "photo": [
                {"file_id": "AAA", "file_unique_id": "A", "width": 90, "height": 90},
            ],
        },
        ptb_bot,
    )


@pytest.fixture
def message_photo_with_caption(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 709,
            "date": 1700000010,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "caption": "look at this cat",
            "photo": [
                {"file_id": "S1", "file_unique_id": "s1", "width": 90, "height": 90},
                {"file_id": "L1", "file_unique_id": "l1", "width": 1280, "height": 1280},
            ],
        },
        ptb_bot,
    )


@pytest.fixture
def message_voice(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 710,
            "date": 1700000011,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "voice": {
                "file_id": "VOICE-A",
                "file_unique_id": "voice-a",
                "duration": 4,
                "mime_type": "audio/ogg",
            },
        },
        ptb_bot,
    )


@pytest.fixture
def message_document(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 711,
            "date": 1700000012,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "document": {
                "file_id": "DOC-A",
                "file_unique_id": "doc-a",
                "file_name": "report.pdf",
                "mime_type": "application/pdf",
            },
        },
        ptb_bot,
    )


@pytest.fixture
def message_channel_post_no_sender(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 808,
            "date": 1700000007,
            "chat": {"id": -1009999999999, "type": "channel", "title": "News"},
            "text": "announcement",
        },
        ptb_bot,
    )


@pytest.fixture
def message_static_sticker(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 720,
            "date": 1700000020,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "sticker": {
                "file_id": "STATIC-STICKER",
                "file_unique_id": "ss",
                "type": "regular",
                "width": 512,
                "height": 512,
                "is_animated": False,
                "is_video": False,
                "emoji": "👍",
            },
        },
        ptb_bot,
    )


@pytest.fixture
def message_video_sticker(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 721,
            "date": 1700000021,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "sticker": {
                "file_id": "VIDEO-STICKER",
                "file_unique_id": "vs",
                "type": "regular",
                "width": 512,
                "height": 512,
                "is_animated": False,
                "is_video": True,
                "emoji": "🎉",
            },
        },
        ptb_bot,
    )


@pytest.fixture
def message_animation(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 722,
            "date": 1700000022,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "animation": {
                "file_id": "ANIM-A",
                "file_unique_id": "anim-a",
                "width": 320,
                "height": 240,
                "duration": 3,
                "mime_type": "video/mp4",
                "file_name": "fun.mp4",
            },
            # Telegram also stuffs animation into the document field.
            "document": {
                "file_id": "ANIM-A",
                "file_unique_id": "anim-a",
                "file_name": "fun.mp4",
                "mime_type": "video/mp4",
            },
        },
        ptb_bot,
    )


@pytest.fixture
def message_video_note(ptb_bot: Bot) -> Message:
    return _make_message(
        {
            "message_id": 723,
            "date": 1700000023,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "video_note": {
                "file_id": "VN-A",
                "file_unique_id": "vn-a",
                "length": 240,
                "duration": 5,
            },
        },
        ptb_bot,
    )


@pytest.fixture
def message_edited(ptb_bot: Bot) -> Message:
    """Edited DM text — same message_id as the original, with edit_date set."""
    return _make_message(
        {
            "message_id": 730,
            "date": 1700000030,
            "edit_date": 1700000040,
            "chat": {"id": 123456789, "type": "private"},
            "from": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
            "text": "fixed typo",
        },
        ptb_bot,
    )


@pytest.fixture
def reaction_added(ptb_bot: Bot) -> Any:
    """User added a 👍 reaction to a bot message (no prior reaction)."""
    from telegram import MessageReactionUpdated

    payload: dict[str, Any] = {
        "chat": {"id": 123456789, "type": "private"},
        "message_id": 901,
        "user": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
        "date": 1700000050,
        "old_reaction": [],
        "new_reaction": [{"type": "emoji", "emoji": "👍"}],
    }
    out = MessageReactionUpdated.de_json(payload, ptb_bot)
    assert out is not None
    return out


@pytest.fixture
def reaction_removed(ptb_bot: Bot) -> Any:
    """User cleared a 👍 reaction (had one before, now none)."""
    from telegram import MessageReactionUpdated

    payload: dict[str, Any] = {
        "chat": {"id": 123456789, "type": "private"},
        "message_id": 901,
        "user": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
        "date": 1700000051,
        "old_reaction": [{"type": "emoji", "emoji": "👍"}],
        "new_reaction": [],
    }
    out = MessageReactionUpdated.de_json(payload, ptb_bot)
    assert out is not None
    return out


@pytest.fixture
def reaction_anonymous_in_supergroup(ptb_bot: Bot) -> Any:
    """Anonymous reaction (actor_chat instead of user) — should be dropped."""
    from telegram import MessageReactionUpdated

    payload: dict[str, Any] = {
        "chat": {"id": -1001234567890, "type": "supergroup", "title": "Big"},
        "message_id": 902,
        "actor_chat": {"id": -1001234567890, "type": "supergroup", "title": "Big"},
        "date": 1700000052,
        "old_reaction": [],
        "new_reaction": [{"type": "emoji", "emoji": "🔥"}],
    }
    out = MessageReactionUpdated.de_json(payload, ptb_bot)
    assert out is not None
    return out


@pytest.fixture
def reaction_custom_emoji_only(ptb_bot: Bot) -> Any:
    """Reaction is a custom (premium) emoji only — should be dropped."""
    from telegram import MessageReactionUpdated

    payload: dict[str, Any] = {
        "chat": {"id": 123456789, "type": "private"},
        "message_id": 903,
        "user": {"id": 123456789, "is_bot": False, "first_name": "Alice"},
        "date": 1700000053,
        "old_reaction": [],
        "new_reaction": [{"type": "custom_emoji", "custom_emoji_id": "5123"}],
    }
    out = MessageReactionUpdated.de_json(payload, ptb_bot)
    assert out is not None
    return out
