"""Unit coverage for ``aios_signal.mentions``."""

from __future__ import annotations

from aios_signal.mentions import build_mention_strings, encode_mentions
from aios_signal.parse import MENTION_PLACEHOLDER

ALICE = "fb2c91e2-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
BOB = "22334455-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
CAROL = "fb2c91e3-cccc-cccc-cccc-cccccccccccc"  # shares 7-char prefix with ALICE


# ── encode_mentions ──────────────────────────────────────────────────


def test_no_members_returns_text_unchanged() -> None:
    text, ordered = encode_mentions("hi @fb2c91e2 ping", member_uuids=[])
    assert text == "hi @fb2c91e2 ping"
    assert ordered == []


def test_no_at_sign_short_circuits() -> None:
    text, ordered = encode_mentions("plain text", member_uuids=[ALICE, BOB])
    assert text == "plain text"
    assert ordered == []


def test_resolves_unique_8char_prefix() -> None:
    text, ordered = encode_mentions("hi @fb2c91e2 ping", member_uuids=[ALICE, BOB])
    assert text == f"hi {MENTION_PLACEHOLDER} ping"
    assert ordered == [ALICE]


def test_resolves_full_uuid() -> None:
    text, ordered = encode_mentions(f"yo @{ALICE}!", member_uuids=[ALICE, BOB])
    assert text == f"yo {MENTION_PLACEHOLDER}!"
    assert ordered == [ALICE]


def test_too_short_prefix_left_alone() -> None:
    # 7 chars below the 8-char minimum — regex doesn't match at all.
    text, ordered = encode_mentions("ping @fb2c91e", member_uuids=[ALICE, CAROL])
    assert text == "ping @fb2c91e"
    assert ordered == []


def test_ambiguous_8char_prefix_returns_no_resolution() -> None:
    # Both DAVE and EVE share the leading "abcd1234" — resolver can't pick
    # one, so the candidate falls through as plain text.
    dave = "abcd1234-dddd-dddd-dddd-dddddddddddd"
    eve = "abcd1234-eeee-eeee-eeee-eeeeeeeeeeee"
    text, ordered = encode_mentions("yo @abcd1234 ping", member_uuids=[dave, eve])
    assert text == "yo @abcd1234 ping"
    assert ordered == []


def test_unresolved_prefix_passes_through() -> None:
    # 8 hex chars but matches no member — leave as plain text.
    text, ordered = encode_mentions("hello @deadbeef", member_uuids=[ALICE, BOB])
    assert text == "hello @deadbeef"
    assert ordered == []


def test_multiple_mentions_preserve_left_to_right_order() -> None:
    text, ordered = encode_mentions("hi @fb2c91e2 and @22334455", member_uuids=[ALICE, BOB])
    assert text == f"hi {MENTION_PLACEHOLDER} and {MENTION_PLACEHOLDER}"
    assert ordered == [ALICE, BOB]


def test_mention_resolution_is_dash_insensitive() -> None:
    text, ordered = encode_mentions("hi @fb2c91e2aaaa", member_uuids=[ALICE, BOB])
    assert MENTION_PLACEHOLDER in text
    assert ordered == [ALICE]


def test_pre_existing_placeholder_stripped_before_encoding() -> None:
    # Forwarded inbound text could carry a stray U+FFFC; if encode_mentions
    # let it through, build_mention_strings would crash on the orphan.
    text, ordered = encode_mentions(
        f"{MENTION_PLACEHOLDER}stray @fb2c91e2", member_uuids=[ALICE, BOB]
    )
    assert text == f"stray {MENTION_PLACEHOLDER}"
    assert ordered == [ALICE]


# ── build_mention_strings ───────────────────────────────────────────


def test_build_strings_returns_empty_when_no_uuids() -> None:
    assert build_mention_strings("plain text", []) == []


def test_build_strings_pairs_placeholders_with_uuids_in_order() -> None:
    msg = f"hi {MENTION_PLACEHOLDER} and {MENTION_PLACEHOLDER}"
    assert build_mention_strings(msg, [ALICE, BOB]) == [
        f"3:1:{ALICE}",
        f"9:1:{BOB}",
    ]


def test_build_strings_uses_utf16_offsets_after_emoji() -> None:
    # 🎉 is one Python code point but two UTF-16 code units — the
    # placeholder must report offset 3, not 2.
    msg = f"🎉 {MENTION_PLACEHOLDER}"
    assert build_mention_strings(msg, [ALICE]) == [f"3:1:{ALICE}"]
