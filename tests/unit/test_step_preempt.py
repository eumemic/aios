"""Unit tests for the #253 preemption plumbing in ``harness/loop.py``.

Pure asyncio + fake pool — the floored predicate's SQL behavior is pinned
against real Postgres in ``tests/e2e/test_preempt_predicate.py``; here we pin
the race orchestration (``_race_model_against_preempt``), the watcher's poll
gate (``_wait_for_preempt``), and the starvation guard (``_preempt_allowed``).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import loop as loop_mod
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.loop import (
    _preempt_allowed,
    _Preempted,
    _race_model_against_preempt,
    _wait_for_preempt,
)
from aios.harness.sweep import find_sessions_needing_inference

SID = "sess_preempt"


def _fake_pool(conn: Any) -> Any:
    pool = MagicMock()

    @asynccontextmanager
    async def _acquire() -> Any:
        yield conn

    pool.acquire = _acquire
    return pool


# ─── _race_model_against_preempt ──────────────────────────────────────────────


class TestRace:
    async def test_model_wins_returns_response_and_cancels_watcher(self) -> None:
        response = object()

        async def model() -> Any:
            return response

        watcher_cancelled = asyncio.Event()

        async def watcher(*args: Any, **kwargs: Any) -> int | None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                watcher_cancelled.set()
                raise
            return None

        with patch.object(loop_mod, "_wait_for_preempt", watcher):
            result = await _race_model_against_preempt(
                model, MagicMock(), InflightToolRegistry(), SID, reacted_floor=0
            )
        assert result is response
        assert watcher_cancelled.is_set()

    async def test_model_error_propagates_after_watcher_teardown(self) -> None:
        async def model() -> Any:
            raise RuntimeError("provider exploded")

        async def watcher(*args: Any, **kwargs: Any) -> int | None:
            await asyncio.sleep(3600)
            return None

        with (
            patch.object(loop_mod, "_wait_for_preempt", watcher),
            pytest.raises(RuntimeError, match="provider exploded"),
        ):
            await _race_model_against_preempt(
                model, MagicMock(), InflightToolRegistry(), SID, reacted_floor=0
            )

    async def test_watcher_wins_cancels_model_and_returns_preempted(self) -> None:
        model_cancelled = asyncio.Event()

        async def model() -> Any:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                model_cancelled.set()
                raise

        async def watcher(*args: Any, **kwargs: Any) -> int | None:
            return 42

        with patch.object(loop_mod, "_wait_for_preempt", watcher):
            result = await _race_model_against_preempt(
                model, MagicMock(), InflightToolRegistry(), SID, reacted_floor=0
            )
        assert result == _Preempted(cancelled_by=42)
        assert model_cancelled.is_set()

    async def test_watcher_failure_degrades_to_wait(self) -> None:
        """Preemption plumbing must never break inference: a dead watcher
        means the step just awaits the model as if policy were wait."""
        response = object()
        model_started = asyncio.Event()

        async def model() -> Any:
            model_started.set()
            await asyncio.sleep(0.01)
            return response

        async def watcher(*args: Any, **kwargs: Any) -> int | None:
            await model_started.wait()
            raise RuntimeError("watcher exploded")

        with patch.object(loop_mod, "_wait_for_preempt", watcher):
            result = await _race_model_against_preempt(
                model, MagicMock(), InflightToolRegistry(), SID, reacted_floor=0
            )
        assert result is response

    async def test_outer_cancellation_tears_down_both_and_propagates(self) -> None:
        """Operator interrupt / step timeout / shutdown cancel the STEP task;
        the race must cancel both children and re-raise — today's semantics."""
        model_cancelled = asyncio.Event()
        watcher_cancelled = asyncio.Event()
        race_running = asyncio.Event()

        async def model() -> Any:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                model_cancelled.set()
                raise

        async def watcher(*args: Any, **kwargs: Any) -> int | None:
            race_running.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                watcher_cancelled.set()
                raise
            return None

        with patch.object(loop_mod, "_wait_for_preempt", watcher):
            race = asyncio.create_task(
                _race_model_against_preempt(
                    model, MagicMock(), InflightToolRegistry(), SID, reacted_floor=0
                )
            )
            await asyncio.wait_for(race_running.wait(), timeout=5)
            race.cancel()
            with pytest.raises(asyncio.CancelledError):
                await race
        assert model_cancelled.is_set()
        assert watcher_cancelled.is_set()


# ─── _wait_for_preempt ────────────────────────────────────────────────────────


class TestWaitForPreempt:
    async def test_first_tick_runs_full_predicate_and_returns_stimulus_seq(self) -> None:
        """The first iteration always runs the full floored predicate — it
        closes the context-build→arm gap (including confirms, which are not
        stimuli and invisible to the watermark)."""
        conn = MagicMock()
        conn.fetchrow = AsyncMock(return_value={"last_event_seq": 5, "last_stimulus_seq": 9})
        conn.fetchval = AsyncMock(return_value=9)
        predicate = AsyncMock(return_value={SID})
        registry = InflightToolRegistry()

        with patch.object(loop_mod, "find_sessions_needing_inference", predicate):
            result = await _wait_for_preempt(_fake_pool(conn), registry, SID, reacted_floor=7)
        assert result == 9
        assert predicate.await_count == 1
        assert predicate.await_args is not None
        assert predicate.await_args.kwargs["session_id"] == SID
        assert predicate.await_args.kwargs["reacted_floor"] == 7

    async def test_returns_none_when_admission_has_no_stimulus_past_floor(self) -> None:
        """A confirmed-dispatch or cancel-marker admission carries no stimulus
        seq past the floor — ``cancelled_by`` is omitted, not fabricated."""
        conn = MagicMock()
        conn.fetchrow = AsyncMock(return_value={"last_event_seq": 5, "last_stimulus_seq": 7})
        conn.fetchval = AsyncMock(return_value=7)
        predicate = AsyncMock(return_value={SID})

        with patch.object(loop_mod, "find_sessions_needing_inference", predicate):
            result = await _wait_for_preempt(
                _fake_pool(conn), InflightToolRegistry(), SID, reacted_floor=7
            )
        assert result is None

    async def test_gate_suppresses_reevaluation_until_last_event_seq_advances(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Steady state is one PK read per tick: the full predicate re-runs
        only when ``last_event_seq`` advances past the value captured BEFORE
        the previous evaluation."""
        monkeypatch.setattr(loop_mod, "_PREEMPT_POLL_INTERVAL_S", 0)
        rows = [{"last_event_seq": 5, "last_stimulus_seq": 3}] * 10 + [
            {"last_event_seq": 6, "last_stimulus_seq": 11}
        ] * 10
        conn = MagicMock()
        conn.fetchrow = AsyncMock(side_effect=rows)
        conn.fetchval = AsyncMock(return_value=11)
        predicate = AsyncMock(side_effect=[set(), {SID}])

        with patch.object(loop_mod, "find_sessions_needing_inference", predicate):
            result = await _wait_for_preempt(
                _fake_pool(conn), InflightToolRegistry(), SID, reacted_floor=3
            )
        assert result == 11
        # Exactly two full evaluations: first tick + the gate advance; the 9
        # unchanged reads in between ran no predicate.
        assert predicate.await_count == 2
        assert conn.fetchrow.await_count == 11


# ─── _preempt_allowed (starvation guard) ──────────────────────────────────────


class TestPreemptAllowed:
    async def _allowed_with_count(self, count: int) -> bool:
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value=count)
        return await _preempt_allowed(_fake_pool(conn), SID, account_id="acc_x")

    async def test_under_cap_allows(self) -> None:
        assert await self._allowed_with_count(loop_mod._PREEMPT_STARVATION_CAP - 1) is True

    async def test_at_cap_blocks(self) -> None:
        assert await self._allowed_with_count(loop_mod._PREEMPT_STARVATION_CAP) is False


# ─── floored-predicate argument contract ──────────────────────────────────────


async def test_reacted_floor_requires_session_id() -> None:
    with pytest.raises(ValueError, match="session_id"):
        await find_sessions_needing_inference(MagicMock(), MagicMock(), reacted_floor=5)
