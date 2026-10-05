"""Unit tests for harness/channels.py.

Pure-function coverage.  After the connector redesign (#200) the
helpers operate on plain ``list[str]`` channel addresses derived from
the event log; the explicit binding/connection structures are gone.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from aios.harness.channels import (
    CHANNEL_NAME_MAX_CHARS,
    apply_monologue_prefix,
    augment_with_focal_paradigm,
    build_focal_paradigm_block,
    channel_display_name,
    max_channels_reminder_local,
    render_channels_reminder,
)
from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT
from aios.harness.tokens import approx_tokens
from aios.models.events import Event


def _user_event(
    seq: int,
    *,
    orig: str | None = None,
    focal_at: str | None = None,
    content: str = "hello",
    metadata: dict[str, Any] | None = None,
) -> Event:
    data: dict[str, Any] = {"role": "user", "content": content}
    if metadata is not None:
        data["metadata"] = metadata
    return Event(
        id=f"evt_{seq:04d}",
        session_id="sess_x",
        seq=seq,
        kind="message",
        data=data,
        cumulative_tokens=None,
        created_at=datetime(2026, 4, 17, tzinfo=UTC),
        orig_channel=orig,
        focal_channel_at_arrival=focal_at,
    )


# ── build_focal_paradigm_block / augment_with_focal_paradigm ───────────────


class TestBuildFocalParadigmBlock:
    """Cache-stable prose introducing the focal-channel model.

    Text does not vary with per-channel state (unread counts, previews) —
    that lives in the channels reminder row.
    """

    def test_no_channels_returns_empty_string(self) -> None:
        assert build_focal_paradigm_block([]) == ""

    def test_mentions_focal_and_switch_channel(self) -> None:
        block = build_focal_paradigm_block(["signal/alice/chat-1"])
        assert "focal" in block.lower()
        assert "switch_channel" in block

    def test_describes_tail_symbols(self) -> None:
        block = build_focal_paradigm_block(["signal/alice/chat-1"])
        assert "▸" in block
        assert "○" in block

    def test_contains_monologue_reminder(self) -> None:
        block = build_focal_paradigm_block(["signal/alice/chat-1"])
        assert "INTERNAL_MONOLOGUE" in block

    def test_timing_prose_requires_output_every_step(self) -> None:
        """The ``### Timing`` prose was inverted as part of the empty-turn
        cascade fix: it no longer tells the model that silence/no-response
        is acceptable (literal-minded models took that as license to emit
        empty turns). It now requires output on every step, even when
        nothing is posted to a channel."""
        block = build_focal_paradigm_block(["signal/alice/chat-1"])
        # New wording present.
        assert "Never end a step with empty output" in block
        assert "every step must still" in block
        # Old wording gone — its presence would re-license the empty turn.
        assert "silence is the right choice" not in block
        assert "no obligation to respond" not in block

    def test_no_per_channel_data_leakage(self) -> None:
        """The block must not name any specific bound channel — that's
        the channels listing's job.  Paradigm prose stays cache-stable.
        """
        block = build_focal_paradigm_block(
            [
                "signal/alice/chat-1",
                "slack/workspace/channel/thread",
            ]
        )
        assert "signal/alice/chat-1" not in block
        assert "slack/workspace/channel/thread" not in block

    def test_describes_phone_down_state(self) -> None:
        block = build_focal_paradigm_block(["signal/alice/chat-1"])
        assert "phone down" in block.lower() or "target=null" in block


class TestAugmentWithFocalParadigm:
    def test_no_channels_returns_base_unchanged(self) -> None:
        assert (
            augment_with_focal_paradigm("you are a helpful agent", []) == "you are a helpful agent"
        )

    def test_appends_block_after_base(self) -> None:
        result = augment_with_focal_paradigm("you are helpful", ["signal/a/1"])
        assert result.startswith("you are helpful")
        assert "switch_channel" in result
        assert "\n\n" in result

    def test_empty_base_yields_block_only(self) -> None:
        result = augment_with_focal_paradigm("", ["signal/a/1"])
        assert "switch_channel" in result
        assert not result.startswith("\n")


# ── render_channels_reminder ───────────────────────────────────────────────


class TestRenderChannelsReminder:
    """The bound-channel listing — content of the channels reminder row.

    Pure data — no prose explaining the paradigm (that's the job of
    :func:`build_focal_paradigm_block`, which is cache-stable and lives in
    the system prompt). Returned as plain text: the composer decides
    whether a row is written (``aios.harness.reminders``).
    """

    _ALICE = "signal/bot/alice"
    _FAMILY = "signal/bot/family"

    def test_no_channels_returns_none(self) -> None:
        assert render_channels_reminder([], [], focal_channel=None) is None

    def test_focal_line_marked_with_triangle_no_unread_count(self) -> None:
        content = render_channels_reminder([self._ALICE], [], focal_channel=self._ALICE)
        assert content is not None
        assert "▸ channel_id=signal/bot/alice (focal)" in content
        focal_line = next(ln for ln in content.splitlines() if "▸" in ln)
        assert not any(ch.isdigit() for ch in focal_line), focal_line

    def test_non_focal_channel_shows_unread_count(self) -> None:
        events = [
            _user_event(1, orig=self._FAMILY, focal_at=self._ALICE, content="hi from mom"),
            _user_event(2, orig=self._FAMILY, focal_at=self._ALICE, content="and again"),
        ]
        content = render_channels_reminder(
            [self._ALICE, self._FAMILY],
            events,
            focal_channel=self._ALICE,
        )
        assert content is not None
        assert "○ channel_id=signal/bot/family — 2 unread" in content

    def test_non_focal_preview_truncated(self) -> None:
        long = "x" * 200
        events = [_user_event(1, orig=self._FAMILY, focal_at=self._ALICE, content=long)]
        content = render_channels_reminder(
            [self._ALICE, self._FAMILY],
            events,
            focal_channel=self._ALICE,
        )
        assert content is not None
        assert "x" * 60 + "…" in content
        assert "x" * 61 not in content

    def test_phone_down_shows_no_focal_marker(self) -> None:
        content = render_channels_reminder(
            [self._ALICE, self._FAMILY],
            [],
            focal_channel=None,
        )
        assert content is not None
        assert "▸" not in content
        assert "(focal)" not in content

    def test_render_is_plain_text_starting_with_the_header(self) -> None:
        content = render_channels_reminder([self._ALICE], [], focal_channel=self._ALICE)
        assert isinstance(content, str)
        assert content.startswith("━━━ Channels ━━━\n")

    def test_zero_unread_non_focal_still_listed(self) -> None:
        content = render_channels_reminder(
            [self._ALICE, self._FAMILY],
            [],
            focal_channel=self._ALICE,
        )
        assert content is not None
        assert f"○ channel_id={self._FAMILY} — 0 unread" in content


class TestChannelDisplayName:
    """#118 type-directed naming ladder: group -> chat_name, dm ->
    sender_name, neither -> None."""

    def test_group_uses_chat_name(self) -> None:
        md = {"chat_type": "group", "chat_name": "AI Bros", "sender_name": "Tom"}
        assert channel_display_name(md, "signal/bot/grp") == "AI Bros"

    def test_dm_uses_sender_name(self) -> None:
        md = {"chat_type": "dm", "chat_name": "ignored", "sender_name": "Tom"}
        assert channel_display_name(md, "signal/bot/x") == "Tom"

    def test_telegram_supergroup_counts_as_group(self) -> None:
        md = {"chat_type": "supergroup", "chat_name": "AI Bros", "sender_name": "Tom"}
        assert channel_display_name(md, "telegram/1/-100") == "AI Bros"

    def test_missing_chat_type_falls_back_to_address_shape(self) -> None:
        md = {"chat_name": "AI Bros", "sender_name": "Tom"}
        assert channel_display_name(md, "telegram/1/-1003881335823") == "AI Bros"
        assert channel_display_name(md, "telegram/1/1595907265") == "Tom"

    def test_group_without_chat_name_has_no_name(self) -> None:
        md = {"chat_type": "group", "sender_name": "Tom"}
        assert channel_display_name(md, "signal/bot/grp") is None

    def test_no_metadata_has_no_name(self) -> None:
        assert channel_display_name(None, "signal/bot/grp") is None
        assert channel_display_name({}, "signal/bot/grp") is None

    def test_blank_and_non_string_names_ignored(self) -> None:
        assert channel_display_name({"chat_type": "group", "chat_name": "  "}, "a/b/c") is None
        assert channel_display_name({"chat_type": "dm", "sender_name": 7}, "a/b/c") is None

    def test_name_normalized_and_truncated(self) -> None:
        md = {"chat_type": "group", "chat_name": "line one\nline two " + "y" * 100}
        name = channel_display_name(md, "a/b/c")
        assert name is not None
        assert "\n" not in name
        assert name == ("line one line two " + "y" * 100)[:CHANNEL_NAME_MAX_CHARS] + "…"


class TestRenderChannelsReminderNames:
    """#118: each listing line carries the latest-observed human-readable
    name after the (still load-bearing) channel_id."""

    _ALICE = "signal/bot/alice"
    _GROUP = "signal/bot/grp"
    _DM = "telegram/bot/1595907265"

    def test_group_name_on_unread_line(self) -> None:
        md = {"chat_type": "group", "chat_name": "AI Bros", "sender_name": "Tom"}
        events = [_user_event(1, orig=self._GROUP, focal_at=self._ALICE, content="yo", metadata=md)]
        content = render_channels_reminder(
            [self._ALICE, self._GROUP], events, focal_channel=self._ALICE
        )
        assert content is not None
        assert f'○ channel_id={self._GROUP} "AI Bros" — 1 unread: "yo"' in content

    def test_dm_name_on_zero_unread_line(self) -> None:
        md = {"chat_type": "dm", "sender_name": "Tom"}
        events = [_user_event(1, orig=self._DM, focal_at=self._DM, content="hi", metadata=md)]
        content = render_channels_reminder(
            [self._ALICE, self._DM], events, focal_channel=self._ALICE
        )
        assert content is not None
        assert f'○ channel_id={self._DM} "Tom" — 0 unread' in content

    def test_focal_line_carries_name(self) -> None:
        md = {"chat_type": "group", "chat_name": "AI Bros"}
        events = [_user_event(1, orig=self._GROUP, focal_at=self._GROUP, metadata=md)]
        content = render_channels_reminder([self._GROUP], events, focal_channel=self._GROUP)
        assert content is not None
        assert f'▸ channel_id={self._GROUP} "AI Bros" (focal)' in content

    def test_latest_observed_name_wins(self) -> None:
        events = [
            _user_event(
                1,
                orig=self._GROUP,
                focal_at=self._ALICE,
                metadata={"chat_type": "group", "chat_name": "Old Name"},
            ),
            _user_event(
                2,
                orig=self._GROUP,
                focal_at=self._ALICE,
                metadata={"chat_type": "group", "chat_name": "New Name"},
            ),
            # A later event with no name doesn't erase the known one.
            _user_event(3, orig=self._GROUP, focal_at=self._ALICE, metadata={"chat_type": "group"}),
        ]
        content = render_channels_reminder(
            [self._ALICE, self._GROUP], events, focal_channel=self._ALICE
        )
        assert content is not None
        assert '"New Name"' in content
        assert "Old Name" not in content

    def test_no_name_line_byte_identical_to_bare(self) -> None:
        events = [
            _user_event(1, orig=self._GROUP, focal_at=self._ALICE, content="yo"),
            _user_event(
                2,
                orig=self._GROUP,
                focal_at=self._ALICE,
                content="yo",
                metadata={"chat_type": "group", "sender_name": "Tom"},
            ),
        ]
        content = render_channels_reminder(
            [self._ALICE, self._GROUP], events, focal_channel=self._ALICE
        )
        assert content == (
            "━━━ Channels ━━━\n"
            f"▸ channel_id={self._ALICE} (focal)\n"
            f'○ channel_id={self._GROUP} — 2 unread: "yo"'
        )

    def test_shared_name_still_disambiguated_by_id(self) -> None:
        other = "signal/bot2/grp"
        md = {"chat_type": "group", "chat_name": "AI Bros"}
        events = [
            _user_event(1, orig=self._GROUP, focal_at=self._ALICE, metadata=md),
            _user_event(2, orig=other, focal_at=self._ALICE, metadata=md),
        ]
        content = render_channels_reminder(
            [self._ALICE, self._GROUP, other], events, focal_channel=self._ALICE
        )
        assert content is not None
        assert f'channel_id={self._GROUP} "AI Bros"' in content
        assert f'channel_id={other} "AI Bros"' in content


class TestMaxChannelsReminderLocal:
    """The reserve must cover the fattest listing a step can WRITE — the
    preview is truncated by code point, so a dense non-ASCII preview costs
    several times an ASCII one of the same length."""

    _ALICE = "signal/bot/alice"
    _OTHERS = tuple(f"signal/bot/peer-{i:02d}" for i in range(6))

    def test_zero_without_channels(self) -> None:
        assert max_channels_reminder_local([]) == 0

    @pytest.mark.parametrize(
        "preview",
        [
            "x" * 200,
            "the quick brown fox jumps over the lazy dog " * 5,
            "日本語のテキストがここに長く続いています" * 6,
            "😀🚀🎉🔥💯🙏" * 20,
            "\U0001f9d1‍\U0001f4bb" * 60,
            "".join(chr(0x1FA70 + i) for i in range(60)),
            "\U0010fffd" * 60,
        ],
    )
    def test_bound_covers_the_real_render(self, preview: str) -> None:
        channels = [self._ALICE, *self._OTHERS]
        # #118: every channel also carries a maximal (over-long, densest)
        # name so the reserve must cover the name clause too.
        md = {"chat_type": "group", "chat_name": "\U0010fffd" * 200}
        events = [
            _user_event(i + 1, orig=addr, focal_at=self._ALICE, content=preview, metadata=md)
            for i, addr in enumerate(self._OTHERS)
        ]
        content = render_channels_reminder(channels, events, focal_channel=self._ALICE)
        assert content is not None
        priced = approx_tokens(
            [
                {"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT},
                {"role": "user", "content": content},
            ]
        )
        assert priced <= max_channels_reminder_local(channels), (preview[:20], priced)


# ── apply_monologue_prefix ─────────────────────────────────────────────────


class TestApplyMonologuePrefix:
    def test_string_content_prefixed(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": "thinking out loud"}
        out = apply_monologue_prefix(msg)
        assert out["content"] == "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: thinking out loud"

    def test_already_prefixed_string_unchanged(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: hi",
        }
        out = apply_monologue_prefix(msg)
        assert out["content"] == "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: hi"

    def test_empty_string_left_alone(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": ""}
        out = apply_monologue_prefix(msg)
        assert out["content"] == ""

    def test_none_content_left_alone(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": []}
        out = apply_monologue_prefix(msg)
        assert out.get("content") is None

    def test_missing_content_left_alone(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "tool_calls": []}
        out = apply_monologue_prefix(msg)
        assert "content" not in out

    def test_list_content_only_first_text_block_prefixed(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "tool_use", "id": "x", "name": "y", "input": {}},
                {"type": "text", "text": "second"},
            ],
        }
        out = apply_monologue_prefix(msg)
        blocks = out["content"]
        assert blocks[0] == {"type": "text", "text": "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: first"}
        assert blocks[1] == {"type": "tool_use", "id": "x", "name": "y", "input": {}}
        assert blocks[2] == {"type": "text", "text": "second"}

    def test_list_content_tool_use_only_left_alone(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "x", "name": "y", "input": {}}],
        }
        out = apply_monologue_prefix(msg)
        assert out["content"] == [{"type": "tool_use", "id": "x", "name": "y", "input": {}}]

    def test_list_content_first_block_already_prefixed_is_idempotent(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: first"},
                {"type": "text", "text": "second"},
            ],
        }
        out = apply_monologue_prefix(msg)
        assert out["content"][0]["text"] == "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: first"
        assert out["content"][1]["text"] == "second"

    def test_returns_new_dict_preserving_other_fields(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [{"id": "x"}],
            "reacting_to": 42,
        }
        out = apply_monologue_prefix(msg)
        assert out["role"] == "assistant"
        assert out["tool_calls"] == [{"id": "x"}]
        assert out["reacting_to"] == 42
        assert msg["content"] == "hi"
