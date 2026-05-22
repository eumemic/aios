"""Tests for the WhatsApp mention encoder."""

from __future__ import annotations

import pytest

from aios_whatsapp.mentions import encode_mentions


@pytest.mark.parametrize(
    ("text", "expected_jids"),
    [
        ("hey @+15551234567 how are you", ["15551234567@s.whatsapp.net"]),
        ("hey @15551234567 how are you", ["15551234567@s.whatsapp.net"]),
        # Multiple mentions, deduped on JID, ordered by first appearance.
        (
            "cc @+15551234567 and @+18007654321 and @15551234567 again",
            ["15551234567@s.whatsapp.net", "18007654321@s.whatsapp.net"],
        ),
        ("no mentions here", []),
    ],
)
def test_encode_mentions_extracts_e164(text: str, expected_jids: list[str]) -> None:
    out_text, jids = encode_mentions(text)
    # Encoder leaves text untouched — WhatsApp's wire format keeps the
    # @<phone> literal in body.
    assert out_text == text
    assert jids == expected_jids


def test_encode_mentions_ignores_email_addresses() -> None:
    # ``user@example.com`` has ``@`` followed by non-digits; the
    # regex's digit-only group rejects it.  Defensive: ``hello@1234``
    # also has letters before the @ which falls outside word boundary.
    _, jids = encode_mentions("ping user@example.com about it")
    assert jids == []
    _, jids = encode_mentions("see https://example.com/page@123 for refs")
    # The @123 is preceded by an alphanumeric "page", so the lookbehind
    # rejects it.
    assert jids == []


def test_encode_mentions_rejects_too_short_or_too_long() -> None:
    # E.164 valid range is 7-15 digits.  Below/above bounds should
    # not be detected as mentions.
    _, jids = encode_mentions("@123")  # 3 digits
    assert jids == []
    _, jids = encode_mentions("@1234567890123456")  # 16 digits
    assert jids == []
    _, jids = encode_mentions("@1234567")  # 7 digits — at lower bound, valid
    assert jids == ["1234567@s.whatsapp.net"]
