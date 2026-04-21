"""Unit tests for ``aios.cli.profile`` — pure-function profiler over span events.

Constructs synthetic span event streams and asserts on the resulting
:class:`Profile`. No HTTP, no CLI — those are exercised separately.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from aios.cli.profile import (
    compute_profile,
    format_duration,
    profile_to_dict,
    render_profile,
)

# ─── span event fixture builder ──────────────────────────────────────────────


class _Stream:
    """Append-order span event builder. Each appended event auto-advances
    ``seq`` and ``created_at`` by ``step_ms`` unless an explicit offset is passed.
    """

    def __init__(self, start: datetime | None = None, step_ms: int = 10) -> None:
        self._t = start or datetime(2026, 4, 21, 12, 0, 0)
        self._seq = 0
        self._step = timedelta(milliseconds=step_ms)
        self.events: list[dict[str, Any]] = []

    def advance(self, ms: int) -> None:
        self._t += timedelta(milliseconds=ms)

    def span(
        self,
        event: str,
        *,
        id: str | None = None,
        advance_ms: int | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        self._seq += 1
        self._t += self._step if advance_ms is None else timedelta(milliseconds=advance_ms)
        ev_id = id if id is not None else f"ev_{self._seq:04d}"
        data = {"event": event, **payload}
        rec = {
            "id": ev_id,
            "seq": self._seq,
            "kind": "span",
            "data": data,
            "created_at": self._t.isoformat(),
        }
        self.events.append(rec)
        return rec


def _full_step(
    s: _Stream,
    *,
    cause: str = "message",
    sweep_ms: int = 5,
    context_build_ms: int = 2,
    model_request_ms: int = 1500,
) -> tuple[str, str]:
    """Emit a complete inside-step span sequence. Returns (step_start_id, step_end_id)."""
    step_start = s.span("step_start", cause=cause)
    sweep_start = s.span("sweep_start", site="entry")
    s.span(
        "sweep_end",
        sweep_start_id=sweep_start["id"],
        repaired_ghosts=0,
        woken_sessions=1,
        advance_ms=sweep_ms,
    )
    cb_start = s.span("context_build_start")
    s.span("context_build_end", context_build_start_id=cb_start["id"], advance_ms=context_build_ms)
    mr_start = s.span("model_request_start")
    s.span("model_request_end", model_request_start_id=mr_start["id"], advance_ms=model_request_ms)
    step_end = s.span("step_end", step_start_id=step_start["id"])
    return step_start["id"], step_end["id"]


def _wasted_wake_step(s: _Stream, *, cause: str) -> None:
    """Emit a step that early-outs at the sweep guard (woken_sessions=0)."""
    step_start = s.span("step_start", cause=cause)
    sweep_start = s.span("sweep_start", site="entry")
    s.span(
        "sweep_end",
        sweep_start_id=sweep_start["id"],
        repaired_ghosts=0,
        woken_sessions=0,
    )
    s.span("step_end", step_start_id=step_start["id"])


def _between_step_work(
    s: _Stream,
    *,
    tool_ms: int = 500,
    tail_sweep_ms: int = 20,
    wake_cause: str = "sweep",
    delay_seconds: int | None = None,
) -> None:
    """Emit the async work that happens between two steps: a tool_execute, a
    tail sweep, and a wake_deferred for the next step."""
    tool_start = s.span("tool_execute_start", tool_name="bash", tool_call_id="tc_1")
    s.span(
        "tool_execute_end",
        tool_execute_start_id=tool_start["id"],
        tool_name="bash",
        tool_call_id="tc_1",
        is_error=False,
        advance_ms=tool_ms,
    )
    sweep_start = s.span("sweep_start", site="tail")
    s.span(
        "sweep_end",
        sweep_start_id=sweep_start["id"],
        repaired_ghosts=0,
        woken_sessions=1,
        advance_ms=tail_sweep_ms,
    )
    payload: dict[str, Any] = {"cause": wake_cause}
    if delay_seconds is not None:
        payload["delay_seconds"] = delay_seconds
    s.span("wake_deferred", **payload)


# ─── tests ───────────────────────────────────────────────────────────────────


class TestInsideStepPairing:
    def test_single_step_groups_all_phases(self) -> None:
        s = _Stream()
        _full_step(s, model_request_ms=1000)

        profile = compute_profile(s.events)

        assert profile.n_steps == 1
        phases = {p.phase: p for p in profile.inside}
        assert set(phases) == {"sweep", "context_build", "model_request"}
        assert phases["model_request"].n == 1
        assert phases["model_request"].total_s == pytest.approx(1.0, abs=0.02)
        # Phase shares sum to approximately 100% of step total.
        total_share = sum(p.share_of_step for p in profile.inside)
        assert 0.9 < total_share < 1.05

    def test_unpaired_end_becomes_warning(self) -> None:
        s = _Stream()
        s.span("step_start", cause="message")
        # Dangling end with unknown start_id.
        s.span("step_end", step_start_id="ev_nonexistent")

        profile = compute_profile(s.events)

        assert profile.n_steps == 0
        assert any("not in window" in w for w in profile.warnings)


class TestTurnSlicing:
    def test_turns_keeps_only_last_n(self) -> None:
        """Three turns of one step each; --turns 1 should keep only the last."""
        s = _Stream()
        _full_step(s, cause="message", model_request_ms=100)
        _between_step_work(s, wake_cause="message")
        _full_step(s, cause="message", model_request_ms=200)
        _between_step_work(s, wake_cause="message")
        _full_step(s, cause="message", model_request_ms=300)

        one = compute_profile(s.events, turns=1)
        all_three = compute_profile(s.events, turns=None)

        assert one.n_steps == 1
        assert all_three.n_steps == 3
        # The last turn's only model_request is the 300ms one.
        mr = next(p for p in one.inside if p.phase == "model_request")
        assert mr.total_s == pytest.approx(0.3, abs=0.02)

    def test_continuation_steps_stay_with_their_turn(self) -> None:
        """cause=sweep is a continuation; --turns 1 on (message, sweep, message) keeps the last message-started turn alone."""
        s = _Stream()
        _full_step(s, cause="message", model_request_ms=100)
        _between_step_work(s, wake_cause="sweep")
        _full_step(s, cause="sweep", model_request_ms=150)  # continuation
        _between_step_work(s, wake_cause="message")
        _full_step(s, cause="message", model_request_ms=200)

        profile = compute_profile(s.events, turns=1)
        assert profile.n_steps == 1  # just the last message-started step

        profile2 = compute_profile(s.events, turns=2)
        assert profile2.n_steps == 3  # last 2 turns = message+continuation, then message


class TestGapClassification:
    def test_same_turn_gap_attributes_tool_and_sweep(self) -> None:
        s = _Stream()
        _full_step(s, cause="message", model_request_ms=100)
        _between_step_work(s, tool_ms=500, tail_sweep_ms=30, wake_cause="sweep")
        _full_step(s, cause="sweep", model_request_ms=50)

        profile = compute_profile(s.events)

        assert profile.n_steps == 2
        same_turn = next((g for g in profile.gaps if g.name == "same-turn"), None)
        assert same_turn is not None
        assert same_turn.n == 1
        assert same_turn.tool_execute_s == pytest.approx(0.5, abs=0.02)
        assert same_turn.sweep_tail_s == pytest.approx(0.03, abs=0.005)

    def test_cross_turn_gap_is_user_idle(self) -> None:
        s = _Stream()
        _full_step(s, cause="message", model_request_ms=50)
        # No async work between turns — just user idle, then wake_deferred(message).
        s.advance(2000)  # 2s of "user idle"
        s.span("wake_deferred", cause="message")
        _full_step(s, cause="message", model_request_ms=50)

        profile = compute_profile(s.events)

        cross = next((g for g in profile.gaps if g.name == "cross-turn"), None)
        assert cross is not None
        assert cross.n == 1
        assert profile.user_idle_s > 1.9
        # Cross-turn gaps should not contribute to orchestration.
        assert profile.orchestration_s == pytest.approx(0.0, abs=0.01)

    def test_earliest_wake_deferred_wins_over_step_cause(self) -> None:
        """Procrastinate coalescing: N wake_deferreds → 1 step_start. The gap's
        category must reflect the EARLIEST wake_deferred.cause, not the step's.
        """
        s = _Stream()
        _full_step(s, cause="message", model_request_ms=50)
        # Sweep wake first (5ms after step_end), then a coalesced message wake 100ms later.
        s.advance(5)
        s.span("wake_deferred", cause="sweep")
        s.advance(100)
        s.span("wake_deferred", cause="message")
        s.advance(5)
        # step_start's own cause disagrees with the earliest wake — this is
        # the coalescing artifact the rule guards against.
        _full_step(s, cause="message", model_request_ms=50)

        profile = compute_profile(s.events)

        assert any(g.name == "same-turn" for g in profile.gaps), (
            "Gap must classify by earliest wake_deferred.cause (sweep), "
            "not by the next step's cause (message)"
        )

    def test_scheduled_gap_records_intentional_delay(self) -> None:
        s = _Stream()
        _full_step(s, cause="message", model_request_ms=50)
        s.advance(100)
        s.span("wake_deferred", cause="scheduled", delay_seconds=30)
        s.advance(200)
        _full_step(s, cause="scheduled", model_request_ms=50)

        profile = compute_profile(s.events)

        scheduled = next((g for g in profile.gaps if g.name == "scheduled/reschedule"), None)
        assert scheduled is not None
        assert scheduled.intentional_delay_s == 30.0


class TestWastedWake:
    def test_wasted_wake_step_marks_woken_zero(self) -> None:
        """A wasted wake is a step whose entry-site sweep returned 0 — the
        guard early-out. The profiler should still see the step, and the
        woken_sessions=0 in the sweep_end is inspectable via JSON mode."""
        s = _Stream()
        _full_step(s, cause="message", model_request_ms=100)
        s.advance(10)
        s.span("wake_deferred", cause="sweep")
        s.advance(5)
        _wasted_wake_step(s, cause="sweep")

        profile = compute_profile(s.events)

        assert profile.n_steps == 2
        sweep_phase = next((p for p in profile.inside if p.phase == "sweep"), None)
        assert sweep_phase is not None
        assert sweep_phase.n == 2  # one entry sweep per step, including the wasted one


class TestRendering:
    def test_render_profile_contains_key_sections(self) -> None:
        s = _Stream()
        _full_step(s, cause="message", model_request_ms=100)

        out = render_profile(compute_profile(s.events))

        assert "INSIDE-STEP" in out
        assert "model_request" in out
        assert "WALL-CLOCK" in out

    def test_render_empty_profile(self) -> None:
        out = render_profile(compute_profile([]))
        assert "no steps" in out

    def test_profile_to_dict_json_roundtrip(self) -> None:
        import json

        s = _Stream()
        _full_step(s, cause="message", model_request_ms=100)

        d = profile_to_dict(compute_profile(s.events))
        # Must be JSON-serializable.
        json.dumps(d)
        assert d["n_steps"] == 1
        assert len(d["inside"]) == 3  # sweep, context_build, model_request


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (1.5, "1.50s"),
            (0.123, "123ms"),
            (0.0005, "500µs"),
            (30.0, "30.00s"),
        ],
    )
    def test_units(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected
