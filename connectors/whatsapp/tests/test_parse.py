"""Tests for parse.py — daemon ``message`` notification → InboundMessage."""

from __future__ import annotations

from typing import Any

from aios_whatsapp.parse import InboundMessage, parse_message


def _dm_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "3EB0BB36C97D4F8C29A4",
        "timestamp_ms": 1700000000000,
        "from_jid": "15553334444@s.whatsapp.net",
        "from_push_name": "Alice",
        "chat_jid": "15553334444@s.whatsapp.net",
        "chat_type": "dm",
        "chat_name": None,
        "is_self": False,
        "text": "hello bot",
    }
    payload.update(overrides)
    return payload


def _group_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "3EB0AAA",
        "timestamp_ms": 1700000001000,
        "from_jid": "15553334444@s.whatsapp.net",
        "from_push_name": "Alice",
        "chat_jid": "111222333@g.us",
        "chat_type": "group",
        "chat_name": "Test Group",
        "is_self": False,
        "text": "group hello",
    }
    payload.update(overrides)
    return payload


def test_parse_message_dm_round_trip() -> None:
    msg = parse_message(_dm_payload())
    assert msg == InboundMessage(
        chat_type="dm",
        chat_jid="15553334444@s.whatsapp.net",
        chat_name=None,
        sender_jid="15553334444@s.whatsapp.net",
        sender_name="Alice",
        message_id="3EB0BB36C97D4F8C29A4",
        timestamp_ms=1700000000000,
        text="hello bot",
    )


def test_parse_message_group_round_trip() -> None:
    msg = parse_message(_group_payload())
    assert msg is not None
    assert msg.chat_type == "group"
    assert msg.chat_jid == "111222333@g.us"
    assert msg.chat_name == "Test Group"
    assert msg.text == "group hello"


def test_parse_message_drops_self_echo() -> None:
    assert parse_message(_dm_payload(is_self=True)) is None


def test_parse_message_drops_empty_text() -> None:
    assert parse_message(_dm_payload(text="")) is None
    assert parse_message(_dm_payload(text=None)) is None


def test_parse_message_drops_broadcast() -> None:
    assert parse_message(_dm_payload(chat_type="broadcast", chat_jid="12345@broadcast")) is None


def test_parse_message_drops_newsletter() -> None:
    assert parse_message(_dm_payload(chat_type="newsletter", chat_jid="99999@newsletter")) is None


def test_parse_message_drops_missing_required_field() -> None:
    # Missing message id
    p = _dm_payload()
    del p["id"]
    assert parse_message(p) is None
    # Missing timestamp
    p = _dm_payload()
    del p["timestamp_ms"]
    assert parse_message(p) is None


def test_parse_message_tolerates_missing_chat_name_for_dm() -> None:
    p = _dm_payload()
    del p["chat_name"]
    msg = parse_message(p)
    assert msg is not None
    assert msg.chat_name is None


def test_parse_message_tolerates_missing_push_name() -> None:
    p = _dm_payload()
    del p["from_push_name"]
    msg = parse_message(p)
    assert msg is not None
    assert msg.sender_name is None
