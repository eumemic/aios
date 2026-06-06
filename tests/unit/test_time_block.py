"""Tests for the per-step current-time tail block."""

from __future__ import annotations

from datetime import UTC, datetime

from aios.harness.time_block import TIME_BLOCK_MAX_LOCAL, TIME_BLOCK_PREFIX, build_time_block
from aios.harness.tokens import approx_tokens


def test_build_time_block_shape_and_format() -> None:
    now = datetime(2026, 6, 6, 19, 5, 49, tzinfo=UTC)  # a Saturday
    block = build_time_block(now)
    # Mirrors the channels tail: a user-role message appended after
    # build_messages (so it never enters the cache-stable prefix).
    assert block["role"] == "user"
    assert block["content"] == (
        "[current time] 2026-06-06 19:05:49 UTC (Saturday) "
        "— system clock, not a message from anyone."
    )
    # The leading marker the cache-breakpoint recognizer keys off.
    assert block["content"].startswith(TIME_BLOCK_PREFIX)


def test_time_block_max_local_is_a_true_upper_bound() -> None:
    # The reserved window budget must never be exceeded by the real block,
    # whatever the weekday — otherwise the composed payload could overshoot
    # window_max. March 2026 spans every weekday several times over.
    for day in range(1, 32):
        now = datetime(2026, 3, day, 23, 59, 59, tzinfo=UTC)
        assert approx_tokens([build_time_block(now)]) <= TIME_BLOCK_MAX_LOCAL
