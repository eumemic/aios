"""Tests for the chunked stable-prefix windowing algorithm.

These cases come directly from the table in the implementation plan
(min=50k, max=150k unless otherwise noted). The point is to verify:

1. Within a chunk, the cutoff is constant — successive turns append to a
   stable prefix.
2. When the cutoff jumps, it jumps by exactly ``(max - min)`` and never
   skips a chunk.
3. The included window stays in the inclusive range ``(min, max]`` while
   the total exceeds max, and equals the total when it fits.
4. The function is monotonic: an event included at total=N is also included
   at total>N (within the same chunk).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aios.harness.window import cumulative_tokens, select_window


@dataclass(frozen=True)
class FakeEvent:
    """Test fixture: an event whose token count is just an attribute."""

    id: int
    tokens: int


def counter(e: FakeEvent) -> int:
    return e.tokens


def make_events(*token_counts: int) -> list[FakeEvent]:
    return [FakeEvent(id=i, tokens=t) for i, t in enumerate(token_counts)]


# ─── basic shape ─────────────────────────────────────────────────────────────


class TestBasicShape:
    def test_empty_input_returns_empty(self) -> None:
        assert select_window([], min_tokens=10, max_tokens=100, token_counter=counter) == []

    def test_under_max_returns_everything(self) -> None:
        events = make_events(10, 20, 30)  # total 60
        result = select_window(events, min_tokens=50, max_tokens=100, token_counter=counter)
        assert result == events

    def test_exactly_max_returns_everything(self) -> None:
        events = make_events(40, 60)  # total 100
        result = select_window(events, min_tokens=50, max_tokens=100, token_counter=counter)
        assert result == events


# ─── snap behavior (the load-bearing case) ───────────────────────────────────


class TestChunkedSnap:
    """Each test corresponds to one row of the table in the plan."""

    def _setup(self, total: int) -> list[FakeEvent]:
        # 1k tokens per event makes the math obvious.
        return make_events(*[1000] * (total // 1000))

    def test_total_100k_fits(self) -> None:
        events = self._setup(100_000)
        result = select_window(
            events,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        assert cumulative_tokens(result, token_counter=counter) == 100_000
        assert result == events

    def test_total_150k_at_max_no_snap(self) -> None:
        events = self._setup(150_000)
        result = select_window(
            events,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        assert cumulative_tokens(result, token_counter=counter) == 150_000
        assert result == events  # everything still fits

    def test_total_151k_first_snap(self) -> None:
        # Use 1k events: 151 events total. First snap drops 100 oldest events.
        events = self._setup(151_000)
        result = select_window(
            events,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        included = cumulative_tokens(result, token_counter=counter)
        assert included == 51_000
        # The 100 oldest got dropped; the 51 newest remain.
        assert result == events[100:]

    def test_total_200k_in_first_chunk(self) -> None:
        events = self._setup(200_000)
        result = select_window(
            events,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        included = cumulative_tokens(result, token_counter=counter)
        assert included == 100_000  # 200 - 100 dropped
        assert result == events[100:]

    def test_total_250k_top_of_first_chunk(self) -> None:
        events = self._setup(250_000)
        result = select_window(
            events,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        included = cumulative_tokens(result, token_counter=counter)
        # Right at the top of the first chunk: included == max
        assert included == 150_000
        assert result == events[100:]

    def test_total_251k_second_snap(self) -> None:
        events = self._setup(251_000)
        result = select_window(
            events,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        included = cumulative_tokens(result, token_counter=counter)
        assert included == 51_000
        # Cutoff jumped to 200k → first 200 events dropped, 51 remain.
        assert result == events[200:]

    def test_total_350k_top_of_second_chunk(self) -> None:
        events = self._setup(350_000)
        result = select_window(
            events,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        included = cumulative_tokens(result, token_counter=counter)
        assert included == 150_000
        assert result == events[200:]

    def test_total_351k_third_snap(self) -> None:
        events = self._setup(351_000)
        result = select_window(
            events,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        included = cumulative_tokens(result, token_counter=counter)
        assert included == 51_000
        assert result == events[300:]


# ─── prefix stability invariant (the whole point) ────────────────────────────


class TestPrefixStability:
    """Verify the cutoff is monotonic non-decreasing as total grows."""

    def test_cutoff_is_monotonic_within_a_chunk(self) -> None:
        # Take two consecutive snapshots of an event log within the same chunk.
        events_at_t1 = make_events(*[1000] * 200)  # total 200k, in first chunk
        events_at_t2 = make_events(*[1000] * 220)  # total 220k, still in first chunk

        window_t1 = select_window(
            events_at_t1,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        window_t2 = select_window(
            events_at_t2,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )

        # The window at t1 is a prefix of the window at t2 (with new events
        # appended to the end). This is what makes the prompt cache hit.
        assert window_t2[: len(window_t1)] == window_t1
        assert len(window_t2) == len(window_t1) + 20  # 20 new events

    def test_cutoff_jumps_at_snap_point(self) -> None:
        # Events at the very edge of the first chunk vs just past the snap.
        events_t1 = make_events(*[1000] * 250)  # total 250k, top of chunk 1
        events_t2 = make_events(*[1000] * 251)  # total 251k, snapped to chunk 2

        window_t1 = select_window(
            events_t1,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        window_t2 = select_window(
            events_t2,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )

        # The prefixes are now different: t2 has snapped, dropping 100 more.
        assert window_t1[0].id != window_t2[0].id
        # Cumulative tokens at t1 was 150k (max); at t2 it's 51k (just above min).
        assert cumulative_tokens(window_t1, token_counter=counter) == 150_000
        assert cumulative_tokens(window_t2, token_counter=counter) == 51_000


# ─── edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_oversized_single_event_overshoots(self) -> None:
        # A single event larger than (max - min) cannot be windowed away.
        # The function still returns something — the safety check lives in
        # the harness, not here.
        events = make_events(200_000)  # one giant event
        result = select_window(
            events,
            min_tokens=50_000,
            max_tokens=150_000,
            token_counter=counter,
        )
        # The included total exceeds max because we can't drop a partial event.
        # Caller is responsible for noticing this and idling the session.
        assert result == events
        assert cumulative_tokens(result, token_counter=counter) == 200_000

    def test_min_must_be_less_than_max(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            select_window(
                make_events(1, 2),
                min_tokens=100,
                max_tokens=100,
                token_counter=counter,
            )
        with pytest.raises(ValueError, match="strictly less than"):
            select_window(
                make_events(1, 2),
                min_tokens=200,
                max_tokens=100,
                token_counter=counter,
            )

    def test_negative_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            select_window(
                make_events(1),
                min_tokens=0,
                max_tokens=10,
                token_counter=counter,
            )
        with pytest.raises(ValueError, match="must be positive"):
            select_window(
                make_events(1),
                min_tokens=10,
                max_tokens=-5,
                token_counter=counter,
            )
