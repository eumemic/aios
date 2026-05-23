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


def test_encode_mentions_accepts_italic_wrapped() -> None:
    # Pre-fix: the lookbehind treated ``_`` as a word char and
    # silently rejected italic-wrapped mentions.  Post-fix: the
    # explicit ASCII boundary excludes ``_``, so ``_@<phone>_``
    # round-trips.
    _, jids = encode_mentions("tell _@+15551234567_ to read this")
    assert jids == ["15551234567@s.whatsapp.net"]


def test_encode_mentions_accepts_full_phone_jid() -> None:
    # Model may copy the raw JID out of an inbound mention block
    # and write it verbatim; the encoder should pass it through
    # untouched.
    _, jids = encode_mentions("cc @15551234567@s.whatsapp.net")
    assert jids == ["15551234567@s.whatsapp.net"]


def test_encode_mentions_accepts_lid_jid() -> None:
    # Round-trip path for peers addressed by LID — the +E.164
    # form doesn't apply, but the raw LID JID should still extract.
    _, jids = encode_mentions("ping @98765@lid please")
    assert jids == ["98765@lid"]


def test_encode_mentions_skips_url_embedded_at_phone() -> None:
    # URLs with @-and-digits in their path or fragment must NOT
    # become accidental mentions.  Adding ``/`` and ``.`` to the
    # explicit boundary class blocks URL-embedded harvesting.
    _, jids = encode_mentions("see https://chat.example/u/@+15551234567 for context")
    assert jids == []
    _, jids = encode_mentions("path: /tags/@15551234567/index")
    assert jids == []


def test_encode_mentions_skips_non_ascii_digits() -> None:
    # Python's ``\\d`` matches Unicode decimal digits by default
    # (Arabic-Indic, Devanagari, fullwidth).  The regex now uses
    # explicit ``[0-9]`` to avoid producing non-ASCII JIDs that
    # WhatsApp would reject.  U+0660 is ARABIC-INDIC DIGIT ZERO;
    # spelled as an escape so the source file stays ASCII-clean.
    arabic_indic_seven_zeros = chr(0x0660) * 7
    _, jids = encode_mentions(f"hey @{arabic_indic_seven_zeros}")
    assert jids == []
