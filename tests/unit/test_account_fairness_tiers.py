"""Per-account fairness gate respects procrastinate priority tiers (#2593 F1).

Procrastinate fetches ``(priority DESC, id ASC)`` and ``defer_wake`` puts workflow
fan-out wakes in the background tier (-10) below foreground (0). The gate's FIFO
must use the same order: a lower-tier waiter must never hold a free slot away
from a higher-tier wake of the same account. These tests drive the REAL gate and,
for the wiring test, the REAL ``run_session_step`` up to the admission decision.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness import runtime
from aios.harness.account_fairness import AccountStepGate
from aios.jobs.app import _BACKGROUND_PRIORITY as BG
from aios.jobs.app import _FOREGROUND_PRIORITY as FG
from tests.unit.conftest import fake_pool_yielding_conn


class _C:
    t = 0.0

    def __call__(self) -> float:
        return self.t


def test_foreground_wake_admitted_past_background_waiter() -> None:
    gate = AccountStepGate(1, stale_after_seconds=30)
    assert gate.try_admit("A", "a_running", priority=FG)
    assert not gate.try_admit("A", "a_bg_child", priority=BG)
    gate.release("A")  # A now has a free slot
    assert gate.try_admit("A", "a_fg_user", priority=FG)
    assert gate.waiting("A") == ["a_bg_child"]  # the bg waiter keeps its place


def test_background_wake_still_queues_behind_foreground_waiter() -> None:
    gate = AccountStepGate(1, stale_after_seconds=30)
    assert gate.try_admit("A", "a_running", priority=FG)
    assert not gate.try_admit("A", "a_fg", priority=FG)
    gate.release("A")
    assert not gate.try_admit("A", "a_bg_new", priority=BG)
    assert gate.try_admit("A", "a_fg", priority=FG)


def test_fifo_within_a_tier_is_unchanged() -> None:
    gate = AccountStepGate(1, stale_after_seconds=30)
    assert gate.try_admit("A", "a1", priority=BG)
    assert not gate.try_admit("A", "bg_old", priority=BG)
    assert not gate.try_admit("A", "fg_old", priority=FG)
    gate.release("A")
    assert not gate.try_admit("A", "fg_new", priority=FG)  # behind fg_old
    assert not gate.try_admit("A", "bg_old", priority=BG)  # behind fg_old (higher tier)
    assert gate.try_admit("A", "fg_old", priority=FG)
    gate.release("A")
    assert not gate.try_admit("A", "bg_old", priority=BG)  # fg_new now ahead
    assert gate.try_admit("A", "fg_new", priority=FG)
    gate.release("A")
    assert gate.try_admit("A", "bg_old", priority=BG)


def test_lower_tier_waiter_is_not_durably_checked_for_higher_tier_wake() -> None:
    clock = _C()
    gate = AccountStepGate(1, stale_after_seconds=10, clock=clock)
    assert gate.try_admit("A", "a1", priority=FG)
    assert not gate.try_admit("A", "bg_ghost", priority=BG)
    gate.release("A")
    clock.t = 100
    assert gate.unconfirmed_waiters_ahead("A", "fg_user", priority=FG) == []
    assert gate.unconfirmed_waiters_ahead("A", "bg_new", priority=BG) == ["bg_ghost"]


def test_single_tenant_workflow_fanout_does_not_delay_own_foreground_message() -> None:
    """One account, no other tenants: 50 background children, then a foreground
    message. Fetch sim = procrastinate (priority DESC, id ASC, scheduled_at)."""
    clock = _C()
    gate = AccountStepGate(2, stale_after_seconds=30, clock=clock)
    ids = itertools.count()
    jobs: list[tuple[int, int, float, str]] = []
    running: list[tuple[float, int, str]] = []
    done: dict[str, float] = {}
    step, backoff, conc = 10.0, 2.0, 4
    prio: dict[str, int] = {}

    def submit(sid: str, p: int, delay: float = 0.0) -> None:
        prio[sid] = p
        jobs.append((p, next(ids), clock.t + delay, sid))

    for i in range(50):
        submit(f"bg{i}", BG)
    t_user, user_in = 1.0, False
    while "fg_user" not in done and clock.t < 10_000:
        if not user_in and clock.t >= t_user:
            submit("fg_user", FG)
            user_in = True
        while len(running) < conc:
            ready = [j for j in jobs if j[2] <= clock.t]
            if not ready:
                break
            j = min(ready, key=lambda j: (-j[0], j[1]))
            jobs.remove(j)
            sid = j[3]
            unc = gate.unconfirmed_waiters_ahead("A", sid, priority=prio[sid])
            if unc:
                q = {x[3] for x in jobs}
                gate.reconcile_waiters("A", checked=unc, live=[s for s in unc if s in q])
            if gate.try_admit("A", sid, priority=prio[sid]):
                heapq.heappush(running, (clock.t + step, j[1], sid))
            else:
                submit(sid, prio[sid], delay=backoff)
        nxt = [r[0] for r in running] + [j[2] for j in jobs if j[2] > clock.t]
        if not user_in:
            nxt.append(t_user)
        clock.t = min(nxt)
        while running and running[0][0] <= clock.t:
            _, _, sid = heapq.heappop(running)
            gate.release("A")
            done[sid] = clock.t
    latency = done["fg_user"] - t_user
    assert latency <= 2 * step + backoff, f"fg message waited {latency}s behind own bg fan-out"


@pytest.fixture
def _gate(monkeypatch: pytest.MonkeyPatch) -> AccountStepGate:
    gate = AccountStepGate(1, stale_after_seconds=30)
    monkeypatch.setattr(runtime, "_account_step_gate", gate)
    return gate


async def _step(session_id: str, *, priority_of: dict[str, int]) -> tuple[AsyncMock, AsyncMock]:
    from aios.harness.loop import _StepResult, run_session_step

    body = AsyncMock(return_value=_StepResult())
    defer = AsyncMock(side_effect=lambda _p, sid, **_k: priority_of[sid])
    pool = fake_pool_yielding_conn(MagicMock())
    prev = runtime.pool
    runtime.pool = pool

    async def _prio(_pool: Any, sid: str) -> int:
        return priority_of[sid]

    try:
        with (
            mock.patch(
                "aios.services.sessions.load_live_session_account_id",
                new=AsyncMock(return_value="acc_A"),
            ),
            mock.patch("aios.harness.loop._run_session_step_body", new=body),
            mock.patch(
                "aios.services.sessions.append_event",
                new=AsyncMock(return_value=MagicMock(id="evt", seq=1)),
            ),
            mock.patch("aios.harness.loop.defer_wake", new=defer),
            mock.patch("aios.harness.loop.wake_priority", new=_prio),
            mock.patch.object(runtime, "require_inflight_tool_registry", return_value=MagicMock()),
        ):
            await run_session_step(session_id, cause="message")
    finally:
        runtime.pool = prev
    return body, defer


async def test_run_session_step_admits_foreground_past_own_background_waiter(
    _gate: AccountStepGate,
) -> None:
    """Wiring: the waiter's tier comes from the tier ``defer_wake`` re-enqueued it
    at; the newcomer's tier is looked up because the decision depends on it."""
    tiers = {"ses_bg": BG, "ses_fg": FG}
    assert _gate.try_admit("acc_A", "ses_running")  # cap=1 held
    body, defer = await _step("ses_bg", priority_of=tiers)
    body.assert_not_awaited()
    defer.assert_awaited_once()
    assert _gate.known_priority("acc_A", "ses_bg") == BG
    _gate.release("acc_A")  # free slot; ses_bg still waiting
    body, defer = await _step("ses_fg", priority_of=tiers)
    body.assert_awaited_once()  # foreground ran; not held for the bg waiter
    defer.assert_not_awaited()
    assert _gate.waiting("acc_A") == ["ses_bg"]
