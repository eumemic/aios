"""Unit tests for focal-channel unread-derivation helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from aios.harness.channels import (
    SWITCH_CHANNEL_METADATA_KEY,
    derive_last_seen,
    derive_unread_counts,
    focal_channel_path,
)
from aios.models.events import Event, EventKind


def _evt(
    seq: int,
    *,
    kind: EventKind = "message",
    role: str | None = "user",
    orig: str | None = None,
    focal_at: str | None = None,
    content: str = "hi",
    data: dict | None = None,
) -> Event:
    if data is None:
        data = {}
        if role is not None:
            data["role"] = role
        if content is not None and role is not None:
            data["content"] = content
    return Event(
        id=f"evt_{seq:04d}",
        session_id="sess_test",
        seq=seq,
        kind=kind,
        data=data,
        cumulative_tokens=None,
        created_at=datetime(2026, 4, 17, tzinfo=UTC),
        orig_channel=orig,
        focal_channel_at_arrival=focal_at,
    )


def _switch_result(
    seq: int,
    target: str | None,
    *,
    success: bool = True,
) -> Event:
    return _evt(
        seq,
        kind="message",
        role=None,
        orig=None,
        focal_at=None,
        data={
            "role": "tool",
            "tool_call_id": f"call_{seq}",
            "content": "switched" if success else "error",
            "metadata": {
                SWITCH_CHANNEL_METADATA_KEY: {
                    "target": target,
                    "success": success,
                }
            },
        },
    )


class TestDeriveLastSeen:
    def test_no_events_returns_zero(self) -> None:
        assert derive_last_seen([], "signal/a/chat") == 0

    def test_no_matching_events_returns_zero(self) -> None:
        events = [_evt(1, orig="signal/a/other", focal_at=None)]
        assert derive_last_seen(events, "signal/a/chat") == 0

    def test_events_while_focal_are_seen(self) -> None:
        """Any user event with focal_at_arrival == X proves the agent was
        focused on X at that seq; therefore all X-origin events up to
        that seq have been visible in the focal view.
        """
        events = [
            _evt(1, orig="signal/a/x", focal_at="signal/a/x"),
            _evt(2, orig="signal/a/x", focal_at="signal/a/x"),
        ]
        assert derive_last_seen(events, "signal/a/x") == 2

    def test_peripheral_focal_still_anchors(self) -> None:
        """Focal=X at arrival, orig=Y — still anchors last_seen_in_X
        (agent was watching X when Y's notification arrived).
        """
        events = [_evt(5, orig="signal/a/y", focal_at="signal/a/x")]
        assert derive_last_seen(events, "signal/a/x") == 5

    def test_successful_switch_anchors_last_seen(self) -> None:
        events = [_switch_result(7, target="signal/a/x", success=True)]
        assert derive_last_seen(events, "signal/a/x") == 7

    def test_failed_switch_not_anchor(self) -> None:
        events = [_switch_result(7, target="signal/a/x", success=False)]
        assert derive_last_seen(events, "signal/a/x") == 0

    def test_switch_to_other_channel_not_anchor(self) -> None:
        events = [_switch_result(7, target="signal/a/other", success=True)]
        assert derive_last_seen(events, "signal/a/x") == 0

    def test_switch_to_none_does_not_anchor_any(self) -> None:
        events = [_switch_result(7, target=None, success=True)]
        assert derive_last_seen(events, "signal/a/x") == 0

    def test_max_of_focal_and_switch_anchors(self) -> None:
        events = [
            _evt(3, orig="signal/a/x", focal_at="signal/a/x"),
            _switch_result(5, target="signal/a/x", success=True),
            _evt(6, orig="signal/a/y", focal_at="signal/a/y"),
        ]
        # Latest anchor for X is the switch at seq 5.
        assert derive_last_seen(events, "signal/a/x") == 5

    def test_legacy_null_events_ignored(self) -> None:
        events = [
            _evt(1, orig=None, focal_at=None),  # pre-migration legacy
            _evt(2, orig="signal/a/x", focal_at="signal/a/x"),
        ]
        assert derive_last_seen(events, "signal/a/x") == 2


class TestDeriveUnreadCounts:
    def test_no_events_all_zero(self) -> None:
        counts = derive_unread_counts([], ["signal/a/x", "signal/a/y"])
        assert counts == {"signal/a/x": 0, "signal/a/y": 0}

    def test_events_while_focal_are_not_unread(self) -> None:
        events = [
            _evt(1, orig="signal/a/x", focal_at="signal/a/x"),
            _evt(2, orig="signal/a/x", focal_at="signal/a/x"),
        ]
        counts = derive_unread_counts(events, ["signal/a/x"])
        assert counts == {"signal/a/x": 0}

    def test_unread_from_peripheral_channels_counted(self) -> None:
        """Events arriving on Y while focal==X count toward unread_in_Y."""
        events = [
            _evt(1, orig="signal/a/x", focal_at="signal/a/x"),
            _evt(2, orig="signal/a/y", focal_at="signal/a/x"),
            _evt(3, orig="signal/a/y", focal_at="signal/a/x"),
        ]
        counts = derive_unread_counts(events, ["signal/a/x", "signal/a/y"])
        assert counts["signal/a/x"] == 0
        assert counts["signal/a/y"] == 2

    def test_switch_resets_unread_for_target(self) -> None:
        """A successful switch to Y is a last_seen anchor for Y — prior
        Y-origin events are no longer unread.
        """
        events = [
            _evt(1, orig="signal/a/y", focal_at="signal/a/x"),
            _evt(2, orig="signal/a/y", focal_at="signal/a/x"),
            _switch_result(3, target="signal/a/y", success=True),
        ]
        counts = derive_unread_counts(events, ["signal/a/y"])
        assert counts == {"signal/a/y": 0}

    def test_unread_after_switch_counted(self) -> None:
        """New Y-origin events arriving after the switch to Y are seen
        (focal=Y at arrival) — still not unread.  But Y-origin events
        arriving with focal != Y after a later switch-away are unread.
        """
        events = [
            _switch_result(1, target="signal/a/y", success=True),
            _evt(2, orig="signal/a/y", focal_at="signal/a/y"),  # seen focally
            _switch_result(3, target="signal/a/x", success=True),
            _evt(4, orig="signal/a/y", focal_at="signal/a/x"),  # unread
            _evt(5, orig="signal/a/y", focal_at="signal/a/x"),  # unread
        ]
        counts = derive_unread_counts(events, ["signal/a/y"])
        assert counts == {"signal/a/y": 2}

    def test_legacy_null_events_not_counted(self) -> None:
        events = [
            _evt(1, orig=None, focal_at=None),
            _evt(2, orig=None, focal_at=None),
            _evt(3, orig="signal/a/x", focal_at=None),  # real peripheral
        ]
        counts = derive_unread_counts(events, ["signal/a/x"])
        assert counts == {"signal/a/x": 1}

    def test_failed_switch_does_not_reset_unread(self) -> None:
        events = [
            _evt(1, orig="signal/a/y", focal_at="signal/a/x"),
            _switch_result(2, target="signal/a/y", success=False),
            _evt(3, orig="signal/a/y", focal_at="signal/a/x"),
        ]
        counts = derive_unread_counts(events, ["signal/a/y"])
        assert counts == {"signal/a/y": 2}

    def test_counts_per_channel_independent(self) -> None:
        events = [
            _evt(1, orig="signal/a/x", focal_at="signal/a/z"),
            _evt(2, orig="signal/a/y", focal_at="signal/a/z"),
            _evt(3, orig="signal/a/y", focal_at="signal/a/z"),
        ]
        counts = derive_unread_counts(events, ["signal/a/x", "signal/a/y", "signal/a/z"])
        assert counts == {"signal/a/x": 1, "signal/a/y": 2, "signal/a/z": 0}

    def test_non_message_events_not_counted(self) -> None:
        events = [
            _evt(1, kind="span", role=None, orig=None, focal_at=None, data={"event": "ping"}),
            _evt(2, orig="signal/a/x", focal_at=None),
        ]
        counts = derive_unread_counts(events, ["signal/a/x"])
        assert counts == {"signal/a/x": 1}

    def test_assistant_events_not_counted_as_unread(self) -> None:
        """Assistant-role events don't get orig_channel stamped (per
        slice 2's contract: orig is derived from inbound metadata).
        Even if they somehow carried orig=X, they're not user stimuli
        and shouldn't count toward unread.
        """
        events = [
            _evt(1, role="assistant", orig=None, focal_at=None, data={"role": "assistant"}),
            _evt(2, orig="signal/a/x", focal_at=None),
        ]
        counts = derive_unread_counts(events, ["signal/a/x"])
        assert counts == {"signal/a/x": 1}


class TestFocalChannelPath:
    """Suffix-extraction helper for MCP _meta injection (slice 6)."""

    def test_three_segment_address(self) -> None:
        # Signal shape: signal/<bot>/<chat_id>
        assert focal_channel_path("signal/bot/alice") == "alice"

    def test_four_segment_address_preserves_inner_slashes(self) -> None:
        # Telegram forum-thread shape: telegram/<bot>/<chat>/<thread>
        assert focal_channel_path("telegram/bot/chat/thread") == "chat/thread"

    def test_deep_address_preserves_all_tail_segments(self) -> None:
        # Defensive: connectors may use arbitrarily deep suffixes.
        assert focal_channel_path("x/y/a/b/c") == "a/b/c"

    def test_none_returns_none(self) -> None:
        assert focal_channel_path(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert focal_channel_path("") is None

    def test_malformed_two_segments_returns_none(self) -> None:
        # Missing suffix — should not leak a garbled value.
        assert focal_channel_path("signal/bot") is None

    def test_trailing_slash_yields_empty_suffix_is_none(self) -> None:
        # "signal/bot/" → 3 segments but suffix is empty → None.
        assert focal_channel_path("signal/bot/") is None
