"""Tests for the Telegram MCP inbound broker."""

from __future__ import annotations

from typing import Any

import pytest

from aios_telegram.inbound import (
    NOTIFICATION_CHANNELS_DELTA,
    NOTIFICATION_CHANNELS_SNAPSHOT,
    NOTIFICATION_MESSAGE,
    NOTIFICATION_REPLAY_LOST,
    TelegramInboundBroker,
    telegram_event_id,
)


class RecordingSession:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def send_message(self, message: Any) -> None:
        self.messages.append(message)


def _notification(message: Any) -> Any:
    return message.message.root


def test_telegram_event_id_is_deterministic() -> None:
    metadata = {"sender_id": 123, "message_id": 7, "timestamp_ms": 1700000000000}
    assert telegram_event_id(bot_id=999, path="123", metadata=metadata) == telegram_event_id(
        bot_id=999,
        path="123",
        metadata=metadata,
    )
    assert telegram_event_id(bot_id=999, path="123", metadata=metadata) != telegram_event_id(
        bot_id=999,
        path="123",
        metadata={**metadata, "message_id": 8},
    )


async def test_subscribe_snapshot_delta_and_publish_message() -> None:
    broker = TelegramInboundBroker(
        bot_id=999,
        initial_channels=[{"channel": "999/123", "display_name": "Alice", "metadata": {}}],
    )
    session = RecordingSession()

    result = await broker.subscribe(account_id="999", since_event_id=None, session=session)
    assert result["status"] == "subscribed"
    assert _notification(session.messages[0]).method == NOTIFICATION_CHANNELS_SNAPSHOT

    await broker.post_message(
        path="-987",
        content="hello",
        metadata={
            "chat_type": "group",
            "chat_name": "Friends",
            "sender_id": 123,
            "sender_name": "Alice",
            "message_id": 1,
            "timestamp_ms": 1700000000000,
        },
    )

    methods = [_notification(m).method for m in session.messages]
    assert methods[-2:] == [NOTIFICATION_CHANNELS_DELTA, NOTIFICATION_MESSAGE]
    delta = _notification(session.messages[-2]).params
    assert delta["upserts"][0]["channel"] == "999/-987"
    assert delta["upserts"][0]["display_name"] == "Friends"
    params = _notification(session.messages[-1]).params
    assert params["channel"] == "999/-987"
    assert params["content"] == "hello"
    assert params["metadata"]["sender_id"] == 123


async def test_subscribe_replays_after_cursor() -> None:
    broker = TelegramInboundBroker(bot_id=999)
    first_metadata = {
        "sender_id": 123,
        "message_id": 1,
        "timestamp_ms": 1700000000000,
    }
    await broker.post_message(path="123", content="one", metadata=first_metadata)
    first_id = telegram_event_id(bot_id=999, path="123", metadata=first_metadata)
    await broker.post_message(
        path="123",
        content="two",
        metadata={
            "sender_id": 123,
            "message_id": 2,
            "timestamp_ms": 1700000001000,
        },
    )

    session = RecordingSession()
    await broker.subscribe(account_id="999", since_event_id=first_id, session=session)

    messages = [m for m in session.messages if _notification(m).method == NOTIFICATION_MESSAGE]
    assert len(messages) == 1
    assert _notification(messages[0]).params["content"] == "two"


async def test_subscribe_reports_replay_lost_for_unknown_cursor() -> None:
    broker = TelegramInboundBroker(bot_id=999)
    await broker.post_message(
        path="123",
        content="one",
        metadata={
            "sender_id": 123,
            "message_id": 1,
            "timestamp_ms": 1700000000000,
        },
    )

    session = RecordingSession()
    result = await broker.subscribe(account_id="999", since_event_id="missing", session=session)

    assert result["replay_lost"] is True
    assert NOTIFICATION_REPLAY_LOST in [_notification(m).method for m in session.messages]


async def test_subscribe_rejects_wrong_account() -> None:
    broker = TelegramInboundBroker(bot_id=999)
    with pytest.raises(ValueError, match="unknown Telegram account"):
        await broker.subscribe(account_id="123", since_event_id=None, session=RecordingSession())
