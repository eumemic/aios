"""Tests for addressing.py — URL-safe chat_id round-tripping."""

from __future__ import annotations

import base64
import secrets

import pytest

from aios_signal.addressing import decode_chat_id, encode_chat_id, is_dm_chat_id


def test_dm_round_trip() -> None:
    uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    encoded = encode_chat_id(uuid, "dm")
    assert encoded == uuid  # DMs pass through
    assert decode_chat_id(encoded) == ("dm", uuid)


def test_group_with_plus_in_base64() -> None:
    raw = "abc+def=="  # contains `+`
    encoded = encode_chat_id(raw, "group")
    assert "+" not in encoded
    assert "-" in encoded
    assert decode_chat_id(encoded) == ("group", raw)


def test_group_with_slash_in_base64() -> None:
    raw = "abc/def=="  # contains `/` which would break path segmentation
    encoded = encode_chat_id(raw, "group")
    assert "/" not in encoded
    assert "_" in encoded
    assert decode_chat_id(encoded) == ("group", raw)


def test_group_with_padding_equals() -> None:
    # Real Signal group IDs are 32 random bytes -> 44 chars of base64 with `==`.
    raw_bytes = secrets.token_bytes(32)
    raw_b64 = base64.b64encode(raw_bytes).decode("ascii")
    assert raw_b64.endswith("=")
    encoded = encode_chat_id(raw_b64, "group")
    assert encoded.endswith("=")  # padding preserved
    assert decode_chat_id(encoded) == ("group", raw_b64)


def test_group_with_both_plus_and_slash() -> None:
    raw = "a+b/c+d/e=="
    encoded = encode_chat_id(raw, "group")
    assert "+" not in encoded and "/" not in encoded
    assert decode_chat_id(encoded) == ("group", raw)


def test_is_dm_chat_id() -> None:
    assert is_dm_chat_id("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee") is True
    assert is_dm_chat_id("not-a-uuid") is False
    assert is_dm_chat_id("abc+def==") is False  # group-id shaped
    # Hex must be valid hex digits — 'g' is not.
    assert is_dm_chat_id("gggggggg-bbbb-cccc-dddd-eeeeeeeeeeee") is False


def test_encode_dm_rejects_non_uuid() -> None:
    with pytest.raises(ValueError):
        encode_chat_id("not-a-uuid", "dm")


def test_decode_uppercase_uuid_is_dm() -> None:
    # Signal-cli sometimes emits upper/mixed case UUIDs.
    uuid = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
    assert decode_chat_id(uuid) == ("dm", uuid)
