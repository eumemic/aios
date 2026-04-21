"""Shared test fixtures.

PTB's ``Message.de_json`` needs a ``Bot`` (or close enough) for its date
deserialization to resolve the right tzinfo. We pass a ``MagicMock(spec=Bot)``
with ``defaults=None`` — that's the minimum PTB will accept without opening
a network connection.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from telegram import Bot, Message

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
