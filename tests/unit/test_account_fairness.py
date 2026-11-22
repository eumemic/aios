"""Per-account execution-time wake fairness (issue #418).

Three layers:

* ``AccountStepGate`` unit semantics (cap, FIFO reservation, stale lapse,
  memory hygiene).
* A deterministic discrete-event simulation of procrastinate's fetch loop
  (``worker_concurrency`` slots, fetch = oldest runnable job, scheduled-in-the-
  future jobs skipped) driving the REAL gate — the issue's acceptance tests:
  (a) a light account's single wake finishes within bounded time behind a
  50-wake burst from another account; (b) no starvation WITHIN the bursting
  account under continued submission (the #2485 F1 property).
* ``run_session_step`` wiring: a refused wake re-defers non-blockingly before
  ``step_start``; an admitted step releases its slot on every exit path.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.config import Settings, get_settings
from aios.harness import runtime
from aios.harness.account_fairness import AccountStepGate
from tests.unit.conftest import fake_pool_yielding_conn

# ── gate semantics ──────────────────────────────────────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_gate_caps_per_account_and_isolates_accounts() -> None:
    gate = AccountStepGate(2, stale_after_seconds=30)
    assert gate.try_admit("A", "a1")
    assert gate.try_admit("A", "a2")
    assert not gate.try_admit("A", "a3")
    # Another account is unaffected by A being at its cap.
    assert gate.try_admit("B", "b1")
    gate.release("A")
    assert gate.try_admit("A", "a3")


def test_gate_reserves_freed_slot_for_oldest_waiter() -> None:
    gate = AccountStepGate(1, stale_after_seconds=30)
    assert gate.try_admit("A", "a1")
    assert not gate.try_admit("A", "a2")  # a2 queues first
    assert not gate.try_admit("A", "a3")
    gate.release("A")
    # A fresh same-account wake cannot jump the line …
    assert not gate.try_admit("A", "a_new")
    # … nor can a younger waiter …
    assert not gate.try_admit("A", "a3")
    # … the oldest waiter gets the slot.
    assert gate.try_admit("A", "a2")
    assert gate.waiting("A") == ["a3", "a_new"]


def test_gate_stale_waiter_lapses() -> None:
    clock = _Clock()
    gate = AccountStepGate(1, stale_after_seconds=10, clock=clock)
    assert gate.try_admit("A", "a1")
    assert not gate.try_admit("A", "ghost")  # never comes back (archived)
    gate.release("A")
    clock.t = 11
    assert gate.try_admit("A", "a2")


def test_gate_forgets_idle_accounts() -> None:
    gate = AccountStepGate(1, stale_after_seconds=30)
    assert gate.try_admit("A", "a1")
    gate.release("A")
    assert gate.tracked_accounts() == 0
    gate.release("A")  # extra release is a harmless no-op
    assert gate.active("A") == 0


def test_gate_rejects_zero_cap() -> None:
    with pytest.raises(ValueError):
        AccountStepGate(0, stale_after_seconds=1)


# ── settings ────────────────────────────────────────────────────────────────


def test_default_cap_is_half_worker_concurrency() -> None:
    s = Settings(worker_concurrency=8)
    assert s.effective_account_step_concurrency_max == 4
    assert Settings(worker_concurrency=1).effective_account_step_concurrency_max == 1


def test_cap_at_or_above_worker_concurrency_rejected() -> None:
    with pytest.raises(ValueError, match="ACCOUNT_STEP_CONCURRENCY_MAX"):
        Settings(worker_concurrency=4, account_step_concurrency_max=4)
    assert Settings(
        worker_concurrency=4, account_step_concurrency_max=3
    ).effective_account_step_concurrency_max == 3


# ── procrastinate fetch-loop simulation ─────────────────────────────────────


@dataclass
class _Job:
    id: int
    session_id: str
    account_id: str
    run_at: float


class _Sim:
    """Discrete-event model of one procrastinate worker running ``wake_session``.

    Fetch order mirrors procrastinate within one priority tier: the oldest job
    (lowest id) whose ``scheduled_at`` has passed. A refused wake is re-inserted
    as a NEW job (higher id) scheduled ``backoff`` in the future and frees its
    slot immediately — exactly what ``run_session_step`` does via ``defer_wake``.
    """

    def __init__(self, *, concurrency: int, cap: int, step: float, backoff: float) -> None:
        self.clock = _Clock()
        self.gate = AccountStepGate(cap, stale_after_seconds=10 * backoff, clock=self.clock)
        self.concurrency = concurrency
        self.step = step
        self.backoff = backoff
        self._ids = itertools.count()
        self.jobs: list[_Job] = []
        self.running: list[tuple[float, int, str, str]] = []
        self.completed: dict[str, float] = {}
        self.bounces = 0
        self.on_complete: Any = None

    def submit(self, session_id: str, account_id: str, delay: float = 0.0) -> None:
        self.jobs.append(_Job(next(self._ids), session_id, account_id, self.clock.t + delay))

    def _fill(self) -> None:
        while len(self.running) < self.concurrency:
            ready = [j for j in self.jobs if j.run_at <= self.clock.t]
            if not ready:
                return
            job = min(ready, key=lambda j: j.id)
            self.jobs.remove(job)
            if self.gate.try_admit(job.account_id, job.session_id):
                heapq.heappush(
                    self.running,
                    (self.clock.t + self.step, job.id, job.session_id, job.account_id),
                )
            else:
                self.bounces += 1
                self.submit(job.session_id, job.account_id, delay=self.backoff)

    def run_until(self, done: Any, horizon: float) -> None:
        while not done() and self.clock.t <= horizon:
            self._fill()
            nexts = [r[0] for r in self.running] + [
                j.run_at for j in self.jobs if j.run_at > self.clock.t
            ]
            if not nexts:
                return
            self.clock.t = min(nexts)
            while self.running and self.running[0][0] <= self.clock.t:
                _, _, sid, acc = heapq.heappop(self.running)
                self.gate.release(acc)
                self.completed[sid] = self.clock.t
                if self.on_complete is not None:
                    self.on_complete(sid, acc)


def test_light_account_finishes_within_bounded_time_behind_50_wake_burst() -> None:
    """Acceptance (a): B's single wake is not served after all of A's 50."""
    sim = _Sim(concurrency=4, cap=2, step=1.0, backoff=2.0)
    for i in range(50):
        sim.submit(f"a{i}", "A")
    sim.submit("b0", "B")  # enqueued strictly AFTER the whole burst
    sim.run_until(lambda: len(sim.completed) == 51, horizon=1_000)

    assert "b0" in sim.completed
    # Without the gate B would finish at ceil(51/4) = 13 step-units. With it, B
    # runs as soon as a non-A slot is free: within the first backoff window.
    assert sim.completed["b0"] <= sim.backoff + sim.step
    # A's burst still drains completely (throughput limited to its cap).
    assert all(f"a{i}" in sim.completed for i in range(50))
    # Bounded bounce amplification: each surplus wake bounces at most once per
    # backoff window, not once per fetch (no busy-loop).
    windows = max(sim.completed.values()) / sim.backoff + 1
    assert sim.bounces <= 50 * windows


def test_no_starvation_within_bursting_account_under_continued_submission() -> None:
    """Acceptance (b), the #2485 F1 property: A keeps submitting fresh wakes
    forever, yet every one of A's original 50 wakes completes, in bounded time,
    and none is overtaken indefinitely by later submissions."""
    sim = _Sim(concurrency=4, cap=2, step=1.0, backoff=2.0)
    fresh = itertools.count()

    def _resubmit(_sid: str, acc: str) -> None:
        if acc == "A":
            # Two fresh A wakes per completion: demand only ever grows.
            sim.submit(f"fresh{next(fresh)}", "A")
            sim.submit(f"fresh{next(fresh)}", "A")

    sim.on_complete = _resubmit
    for i in range(50):
        sim.submit(f"a{i}", "A")
    originals = {f"a{i}" for i in range(50)}
    sim.run_until(lambda: originals <= sim.completed.keys(), horizon=1_000)

    assert originals <= sim.completed.keys(), "an original A wake starved"
    # cap=2, step=1 → 25 units of A capacity for 50 wakes; allow one backoff
    # window of slack per admission round for the reservation hand-off.
    assert max(sim.completed[s] for s in originals) <= 50 / 2 * (sim.step + sim.backoff)
    # FIFO: fresh wakes (submitted after the burst) never overtake the burst —
    # every fresh wake that completed did so no earlier than the burst's head.
    first_original = min(sim.completed[s] for s in originals)
    assert all(t >= first_original for s, t in sim.completed.items() if s.startswith("fresh"))


# ── run_session_step wiring ─────────────────────────────────────────────────


@pytest.fixture
def _gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[AccountStepGate]:
    gate = AccountStepGate(1, stale_after_seconds=30)
    monkeypatch.setattr(runtime, "_account_step_gate", gate)
    yield gate


def _patched_step(append_event: AsyncMock, defer_wake: AsyncMock, body: Any) -> Any:
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value="acc_A")
    pool = fake_pool_yielding_conn(conn)
    return pool, (
        mock.patch("aios.services.sessions.load_live_session_account_id", new=AsyncMock(return_value="acc_A")),
        mock.patch("aios.harness.loop._run_session_step_body", new=body),
        mock.patch("aios.services.sessions.append_event", new=append_event),
        mock.patch("aios.harness.loop.defer_wake", new=defer_wake),
        mock.patch.object(runtime, "require_inflight_tool_registry", return_value=MagicMock()),
    )


async def test_refused_wake_redefers_without_step_start(_gate: AccountStepGate) -> None:
    from aios.harness.loop import run_session_step

    assert _gate.try_admit("acc_A", "ses_other")  # account already at cap=1
    append_event = AsyncMock(return_value=MagicMock(id="evt", seq=1))
    defer = AsyncMock()
    body = AsyncMock()
    pool, patches = _patched_step(append_event, defer, body)
    prev = runtime.pool
    runtime.pool = pool
    try:
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            await run_session_step("ses_1", cause="message")
    finally:
        runtime.pool = prev

    body.assert_not_awaited()
    append_event.assert_not_awaited()  # no step_start for a bounce
    defer.assert_awaited_once()
    kwargs = defer.await_args.kwargs
    assert kwargs["cause"] == "reschedule"
    assert kwargs["delay_seconds"] == get_settings().account_fairness_backoff_seconds
    assert kwargs["account_id"] == "acc_A"
    assert _gate.active("acc_A") == 1  # the bounce did not consume or free a slot
    assert _gate.waiting("acc_A") == ["ses_1"]


async def test_admitted_step_releases_slot_even_on_error(_gate: AccountStepGate) -> None:
    from aios.harness.loop import run_session_step

    seen_active: list[int] = []

    async def _boom(*_a: Any, **_k: Any) -> Any:
        seen_active.append(_gate.active("acc_A"))
        raise RuntimeError("boom")

    append_event = AsyncMock(return_value=MagicMock(id="evt", seq=1))
    defer = AsyncMock()
    pool, patches = _patched_step(append_event, defer, _boom)
    prev = runtime.pool
    runtime.pool = pool
    try:
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            mock.patch(
                "aios.harness.loop._apply_retry_or_failure", new=AsyncMock(return_value=None)
            ),
            pytest.raises(RuntimeError),
        ):
            await run_session_step("ses_1", cause="message")
    finally:
        runtime.pool = prev

    assert seen_active == [1]  # held while the body ran
    assert _gate.active("acc_A") == 0  # released on the error path
