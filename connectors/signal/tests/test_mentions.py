"""Unit coverage for ``aios_signal.mentions.encode_mentions``."""

from __future__ import annotations

from aios_signal.mentions import encode_mentions
from aios_signal.parse import MENTION_PLACEHOLDER

ALICE = "fb2c91e2-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
BOB = "22334455-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
CAROL = "fb2c91e3-cccc-cccc-cccc-cccccccccccc"  # shares 7-char prefix with ALICE


def test_no_members_returns_text_unchanged() -> None:
    text, mentions = encode_mentions("hi @fb2c91e2 ping", member_uuids=[])
    assert text == "hi @fb2c91e2 ping"
    assert mentions == []


def test_no_at_sign_short_circuits() -> None:
    text, mentions = encode_mentions("plain text", member_uuids=[ALICE, BOB])
    assert text == "plain text"
    assert mentions == []


def test_resolves_unique_8char_prefix() -> None:
    text, mentions = encode_mentions("hi @fb2c91e2 ping", member_uuids=[ALICE, BOB])
    assert text == f"hi {MENTION_PLACEHOLDER} ping"
    assert mentions == [f"3:1:{ALICE}"]


def test_resolves_full_uuid() -> None:
    text, mentions = encode_mentions(f"yo @{ALICE}!", member_uuids=[ALICE, BOB])
    assert text == f"yo {MENTION_PLACEHOLDER}!"
    assert mentions == [f"3:1:{ALICE}"]


def test_ambiguous_prefix_left_alone() -> None:
    # 7 chars below the 8-char minimum; even if it WERE matched, ALICE and CAROL
    # both share the leading "fb2c91e" prefix so the resolver would refuse.
    text, mentions = encode_mentions("ping @fb2c91e", member_uuids=[ALICE, CAROL])
    assert text == "ping @fb2c91e"
    assert mentions == []


def test_unresolved_prefix_passes_through() -> None:
    # 8 hex chars but matches no member — leave as plain text.
    text, mentions = encode_mentions("hello @deadbeef", member_uuids=[ALICE, BOB])
    assert text == "hello @deadbeef"
    assert mentions == []


def test_multiple_mentions_get_separate_entries() -> None:
    text, mentions = encode_mentions("hi @fb2c91e2 and @22334455", member_uuids=[ALICE, BOB])
    assert text == f"hi {MENTION_PLACEHOLDER} and {MENTION_PLACEHOLDER}"
    assert mentions == [f"3:1:{ALICE}", f"9:1:{BOB}"]


def test_mention_after_emoji_uses_utf16_offset() -> None:
    # 🎉 is one Python char but two UTF-16 code units — the placeholder
    # must be reported at offset 2, not 1.
    text, mentions = encode_mentions("🎉 @fb2c91e2", member_uuids=[ALICE, BOB])
    assert text == f"🎉 {MENTION_PLACEHOLDER}"
    assert mentions == [f"3:1:{ALICE}"]


def test_mention_resolution_is_dash_insensitive() -> None:
    # Strip dashes both sides — agent might use "@fb2c91e2aaaa" without dashes.
    text, mentions = encode_mentions("hi @fb2c91e2aaaa", member_uuids=[ALICE, BOB])
    assert MENTION_PLACEHOLDER in text
    assert mentions == [f"3:1:{ALICE}"]
