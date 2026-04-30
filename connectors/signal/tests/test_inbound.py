"""Tests for the Signal MCP inbound broker."""

from __future__ import annotations

from typing import Any

from aios_signal.daemon import GroupInfo
from aios_signal.inbound import (
    NOTIFICATION_CHANNELS_SNAPSHOT,
    NOTIFICATION_MESSAGE,
    SignalInboundBroker,
    initial_signal_channels,
    signal_event_id,
)


class RecordingSession:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def send_message(self, message: Any) -> None:
        self.messages.append(message)


def _notification(message: Any) -> Any:
    return message.message.root


def test_initial_signal_channels_include_contacts_and_groups() -> None:
    channels = initial_signal_channels(
        bot_uuid="99999999-8888-7777-6666-555555555555",
        contact_names={
            "99999999-8888-7777-6666-555555555555": "Me",
            "11111111-2222-3333-4444-555555555555": "Alice",
            "not-a-uuid": "Ignored",
        },
        groups=[
            GroupInfo(
                id="group==",
                name="Friends",
                member_uuids=["11111111-2222-3333-4444-555555555555"],
            )
        ],
    )

    assert [c["channel"] for c in channels] == [
        "99999999-8888-7777-6666-555555555555/11111111-2222-3333-4444-555555555555",
        "99999999-8888-7777-6666-555555555555/group==",
    ]


def test_signal_event_id_is_deterministic() -> None:
    metadata = {"sender_uuid": "u", "timestamp_ms": 123}
    assert signal_event_id(bot_uuid="b", path="p", metadata=metadata) == signal_event_id(
        bot_uuid="b", path="p", metadata=metadata
    )


async def test_subscribe_snapshot_and_publish_message() -> None:
    broker = SignalInboundBroker(
        bot_uuid="bot",
        initial_channels=[{"channel": "bot/chat", "display_name": "Chat", "metadata": {}}],
    )
    session = RecordingSession()

    result = await broker.subscribe(account_id="bot", since_event_id=None, session=session)
    assert result["status"] == "subscribed"
    assert _notification(session.messages[0]).method == NOTIFICATION_CHANNELS_SNAPSHOT

    await broker.post_message(
        path="chat",
        content="hello",
        metadata={"sender_uuid": "alice", "timestamp_ms": 1, "sender_name": "Alice"},
    )

    methods = [_notification(m).method for m in session.messages]
    assert methods[-1] == NOTIFICATION_MESSAGE
    params = _notification(session.messages[-1]).params
    assert params["channel"] == "bot/chat"
    assert params["content"] == "hello"
    assert params["metadata"]["sender_uuid"] == "alice"


async def test_subscribe_replays_after_cursor() -> None:
    broker = SignalInboundBroker(bot_uuid="bot")
    await broker.post_message(
        path="chat",
        content="one",
        metadata={"sender_uuid": "alice", "timestamp_ms": 1},
    )
    first_id = signal_event_id(
        bot_uuid="bot",
        path="chat",
        metadata={"sender_uuid": "alice", "timestamp_ms": 1},
    )
    await broker.post_message(
        path="chat",
        content="two",
        metadata={"sender_uuid": "alice", "timestamp_ms": 2},
    )

    session = RecordingSession()
    await broker.subscribe(account_id="bot", since_event_id=first_id, session=session)

    messages = [m for m in session.messages if _notification(m).method == NOTIFICATION_MESSAGE]
    assert len(messages) == 1
    assert _notification(messages[0]).params["content"] == "two"

