"""Tests for :func:`build_metadata` — InboundMessage → aios event metadata.

The focus here is the mention-surfacing contract that matches the signal
connector's shape (``metadata.mentions`` list + ``self_mentioned`` bool),
so the harness's per-platform-uniform rendering in ``context.py`` works
identically across telegram and signal inbounds.
"""

from __future__ import annotations

from telegram import Message

from aios_telegram.connector import build_metadata
from aios_telegram.parse import parse_message


def test_no_mentions_omits_keys(message_dm_text: Message, bot_id: int) -> None:
    msg = parse_message(message_dm_text, bot_id=bot_id)
    assert msg is not None
    metadata = build_metadata(msg, bot_id)
    assert "mentions" not in metadata
    assert "self_mentioned" not in metadata


def test_text_mention_of_bot_marks_self_mentioned(
    message_text_mention_of_bot: Message, bot_id: int
) -> None:
    msg = parse_message(message_text_mention_of_bot, bot_id=bot_id)
    assert msg is not None
    metadata = build_metadata(msg, bot_id)
    assert metadata["mentions"] == [{"uuid": str(bot_id), "name": "TestBot"}]
    assert metadata["self_mentioned"] is True


def test_text_mention_of_other_user_does_not_mark_self(
    message_text_mention_of_other_user: Message, bot_id: int
) -> None:
    msg = parse_message(message_text_mention_of_other_user, bot_id=bot_id)
    assert msg is not None
    metadata = build_metadata(msg, bot_id)
    assert metadata["mentions"] == [{"uuid": "444555666", "name": "Carol Doe"}]
    assert metadata["self_mentioned"] is False


def test_plain_username_mention_of_bot_marks_self(
    message_plain_username_mention_of_bot: Message, bot_id: int
) -> None:
    msg = parse_message(
        message_plain_username_mention_of_bot, bot_id=bot_id, bot_username="testbot"
    )
    assert msg is not None
    metadata = build_metadata(msg, bot_id)
    assert metadata["mentions"] == [{"uuid": str(bot_id), "name": "testbot"}]
    assert metadata["self_mentioned"] is True
