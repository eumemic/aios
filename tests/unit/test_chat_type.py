"""Unit tests for the connector-agnostic ``chat_type_of`` derivation (#1613).

``chat_type`` is a pure function of the channel address — UUID/non-negative
numeric ⇒ ``dm``, URL-safe base64/negative numeric ⇒ ``group`` — mirroring
signal ``decode_chat_id``'s UUID-vs-base64 test without importing the connector
package.
"""

from __future__ import annotations

import pytest

from aios.harness.chat_type import chat_type_of

# A real-looking signal ACI UUID (DM) and a URL-safe-base64 group id.
SIGNAL_DM = "6c21718f-f095-483f-8cd6-610137d581aa"
SIGNAL_GROUP = "abcDEF123_-=="


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        # Full channel path (<connector>/<account>/<chat_id>) — signal.
        (f"signal/bot-uuid/{SIGNAL_DM}", "dm"),
        (f"signal/bot-uuid/{SIGNAL_GROUP}", "group"),
        # Bare trailing chat_id segment.
        (SIGNAL_DM, "dm"),
        (SIGNAL_GROUP, "group"),
        # Telegram numeric ids: non-negative ⇒ dm, negative ⇒ group.
        ("telegram/bot/123456789", "dm"),
        ("telegram/bot/-1001234567890", "group"),
        ("123456789", "dm"),
        ("-1001234567890", "group"),
        # No channel / empty / unrecognized ⇒ None (never falsely claims a type).
        (None, None),
        ("", None),
        ("signal/bot/", None),
        ("signal/bot/!!not-base64!!", None),
    ],
)
def test_chat_type_of(channel: str | None, expected: str | None) -> None:
    assert chat_type_of(channel) == expected


def test_matches_signal_decode_chat_id() -> None:
    """``chat_type_of`` must agree with signal ``decode_chat_id``'s classification
    on the canonical signal dm/group shapes (the spec's parity requirement)."""
    # Reproduce decode_chat_id's test without importing the connector package.
    import re

    dm_re = re.compile(
        r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
    )
    b64_re = re.compile(r"\A[A-Za-z0-9_-]+=*\Z")

    def decode_chat_type(chat_id: str) -> str:
        if dm_re.match(chat_id):
            return "dm"
        assert b64_re.match(chat_id)
        return "group"

    for chat_id in (SIGNAL_DM, SIGNAL_GROUP):
        assert chat_type_of(chat_id) == decode_chat_type(chat_id)
        assert chat_type_of(f"signal/bot/{chat_id}") == decode_chat_type(chat_id)
