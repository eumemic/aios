"""Tests for addressing.py — WhatsApp JID validation + chat_type detection."""

from __future__ import annotations

import pytest

from aios_whatsapp.addressing import chat_type_of, is_valid_chat_id, jid_user_segment


def test_is_valid_chat_id_accepts_dm() -> None:
    assert is_valid_chat_id("15551234567@s.whatsapp.net")


def test_is_valid_chat_id_accepts_group() -> None:
    assert is_valid_chat_id("111222333@g.us")


def test_is_valid_chat_id_accepts_broadcast() -> None:
    assert is_valid_chat_id("12345@broadcast")


def test_is_valid_chat_id_accepts_newsletter() -> None:
    assert is_valid_chat_id("99999@newsletter")


def test_is_valid_chat_id_accepts_lid_device_suffix() -> None:
    # Newer LID-routed accounts have a `:<device>` segment.
    assert is_valid_chat_id("15551234567:1@s.whatsapp.net")


def test_is_valid_chat_id_rejects_garbage() -> None:
    assert not is_valid_chat_id("not-a-jid")
    assert not is_valid_chat_id("")
    assert not is_valid_chat_id("12345@unknown.server")
    assert not is_valid_chat_id("@s.whatsapp.net")
    assert not is_valid_chat_id("abc@s.whatsapp.net")  # non-numeric user


def test_chat_type_of_dispatches_by_server_suffix() -> None:
    assert chat_type_of("15551234567@s.whatsapp.net") == "dm"
    assert chat_type_of("111222333@g.us") == "group"
    assert chat_type_of("12345@broadcast") == "broadcast"
    assert chat_type_of("99999@newsletter") == "newsletter"
    assert chat_type_of("15551234567:1@s.whatsapp.net") == "dm"


def test_chat_type_of_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="not a WhatsApp JID"):
        chat_type_of("garbage")


def test_jid_user_segment_extracts_user() -> None:
    assert jid_user_segment("15551234567@s.whatsapp.net") == "15551234567"
    assert jid_user_segment("111222333@g.us") == "111222333"
    # Device suffix is part of the LID identity, not the user segment.
    assert jid_user_segment("15551234567:1@s.whatsapp.net") == "15551234567"
