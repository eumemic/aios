"""An aborted GC tick must be legible to an operator.

The abort itself is CORRECT and is deliberately left alone: when ``_gc_once``
raises, ``pressure_callback`` is never invoked, so ``_provisioning_pressure``
retains its last known value and the cold-provision admission gate keeps
whatever state it had (fail-static). Swallowing the failure and reporting a
fabricated ``0 bytes used`` all-clear would re-open provisioning onto a
possibly-full disk — the aios#2138 inversion relocated to the budget figure.

What the abort did NOT do is tell anyone what it costs: no disk reclaimed, and
an admission gate frozen on a stale figure. These tests pin that reporting, and
pin that the fail-static behaviour itself is unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from aios.sandbox.backends.base import SandboxCapacityError
from aios.sandbox.registry import GcPressureResult, SandboxRegistry
from tests.helpers.sandbox import FakeBackend


async def _run_failing_ticks(
    registry: SandboxRegistry, ticks: int, monkeypatch: pytest.MonkeyPatch
) -> list[GcPressureResult]:
    """Drive ``ticks`` consecutive aborted ticks; return callback invocations."""
    calls: list[GcPressureResult] = []

    async def boom(pool: Any) -> GcPressureResult:
        raise RuntimeError("incomplete managed image enumeration")

    monkeypatch.setattr(registry, "_gc_once", boom)
    # Collapse the hourly sleep so N ticks run inside the test.
    monkeypatch.setattr("aios.sandbox.registry._GC_INTERVAL_SECONDS", 0)

    task = asyncio.create_task(registry._gc_loop(cast(Any, object()), calls.append))
    for _ in range(ticks * 4):
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return calls


async def test_aborted_tick_reports_what_the_failure_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The abort names the carried-forward gate state, not just a traceback."""
    registry = SandboxRegistry(backend=cast(Any, FakeBackend()))
    registry.set_provisioning_pressure(GcPressureResult(pool_used_bytes=99, pool_budget_bytes=10))
    warning = Mock()
    monkeypatch.setattr("aios.sandbox.registry.log.warning", warning)
    monkeypatch.setattr("aios.sandbox.registry.log.exception", Mock())

    await _run_failing_ticks(registry, 1, monkeypatch)

    assert warning.called
    # The FIRST aborted tick (the interval is collapsed to 0, so the loop may
    # have spun again before cancellation).
    first = warning.call_args_list[0]
    event, kwargs = first[0][0], first[1]
    assert event == "sandbox.gc_tick_failed"
    # The operator-relevant consequences of the abort.
    assert kwargs["reclaimed_this_tick"] is False
    assert kwargs["carried_pool_used_bytes"] == 99
    assert kwargs["carried_pool_budget_bytes"] == 10
    assert kwargs["provisioning_gate_closed"] is True
    assert kwargs["consecutive_failures"] == 1


async def test_sustained_failure_escalates_to_exception_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One aborted tick is a hiccup; a run of them is an unreclaimed-disk outage."""
    registry = SandboxRegistry(backend=cast(Any, FakeBackend()))
    warning = Mock()
    exception = Mock()
    monkeypatch.setattr("aios.sandbox.registry.log.warning", warning)
    monkeypatch.setattr("aios.sandbox.registry.log.exception", exception)

    await _run_failing_ticks(registry, 4, monkeypatch)

    assert exception.called, "sustained GC failure never escalated"
    assert exception.call_args[0][0] == "sandbox.gc_tick_failed"
    assert exception.call_args[1]["consecutive_failures"] >= 3


async def test_aborted_tick_still_withholds_the_all_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION FENCE on the fail-static property (the refused 'fix').

    The abort must NOT call ``pressure_callback``: the admission gate stays on
    the last known figure and remains CLOSED. If a future change reports a
    fabricated all-clear on enumeration failure, this goes red.
    """
    registry = SandboxRegistry(backend=cast(Any, FakeBackend()))
    registry.set_provisioning_pressure(GcPressureResult(pool_used_bytes=99, pool_budget_bytes=10))
    monkeypatch.setattr("aios.sandbox.registry.log.warning", Mock())
    monkeypatch.setattr("aios.sandbox.registry.log.exception", Mock())

    calls = await _run_failing_ticks(registry, 3, monkeypatch)

    assert calls == [], "a failed tick published a pressure figure"
    assert registry._provisioning_pressure == GcPressureResult(
        pool_used_bytes=99, pool_budget_bytes=10
    )
    with pytest.raises(SandboxCapacityError, match="snapshot capacity pressure"):
        registry._admit_capacity_provision("sess_new", account_id="acct")


async def test_healthy_tick_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEGATIVE CONTROL: a succeeding tick publishes its pressure as before.

    No failure log, and the callback receives the tick's own figure — the
    observability change must be inert on the healthy path.
    """
    registry = SandboxRegistry(backend=cast(Any, FakeBackend()))
    warning = Mock()
    exception = Mock()
    monkeypatch.setattr("aios.sandbox.registry.log.warning", warning)
    monkeypatch.setattr("aios.sandbox.registry.log.exception", exception)
    monkeypatch.setattr("aios.sandbox.registry._GC_INTERVAL_SECONDS", 0)

    healthy = GcPressureResult(pool_used_bytes=5, pool_budget_bytes=10)

    async def ok(pool: Any) -> GcPressureResult:
        return healthy

    monkeypatch.setattr(registry, "_gc_once", ok)
    calls: list[GcPressureResult] = []

    task = asyncio.create_task(registry._gc_loop(cast(Any, object()), calls.append))
    for _ in range(8):
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert calls and all(c == healthy for c in calls)
    warning.assert_not_called()
    exception.assert_not_called()


async def test_recovery_after_failures_is_announced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GC that comes back must say so, or the outage never visibly ends."""
    registry = SandboxRegistry(backend=cast(Any, FakeBackend()))
    info = Mock()
    monkeypatch.setattr("aios.sandbox.registry.log.info", info)
    monkeypatch.setattr("aios.sandbox.registry.log.warning", Mock())
    monkeypatch.setattr("aios.sandbox.registry.log.exception", Mock())
    monkeypatch.setattr("aios.sandbox.registry._GC_INTERVAL_SECONDS", 0)

    ticks = 0

    async def flaky(pool: Any) -> GcPressureResult:
        nonlocal ticks
        ticks += 1
        if ticks <= 2:
            raise RuntimeError("incomplete managed image enumeration")
        return GcPressureResult()

    monkeypatch.setattr(registry, "_gc_once", flaky)

    task = asyncio.create_task(registry._gc_loop(cast(Any, object()), lambda p: None))
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    recovered = [c for c in info.call_args_list if c[0][0] == "sandbox.gc_tick_recovered"]
    assert recovered, "GC recovery was never reported"
    assert recovered[0][1]["failed_ticks"] == 2


# ── aios#2343: a dead GC must be visible without reading container logs ────


def _fresh_health(registry: SandboxRegistry) -> Any:
    from aios.harness.production_watchdogs import SnapshotGcHealth

    health = SnapshotGcHealth()
    registry._gc_health = health
    return health


async def _drive(
    registry: SandboxRegistry,
    outcomes: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run exactly ``len(outcomes)`` ticks (True = success, False = failure)."""
    monkeypatch.setattr("aios.sandbox.registry._GC_INTERVAL_SECONDS", 0)
    remaining = list(outcomes)
    done = asyncio.Event()

    async def tick(pool: Any) -> GcPressureResult:
        if not remaining:
            done.set()
            await asyncio.Event().wait()
        ok = remaining.pop(0)
        if not ok:
            raise TimeoutError("docker cli timed out after 30.0s")
        return GcPressureResult()

    monkeypatch.setattr(registry, "_gc_once", tick)
    task = asyncio.create_task(registry._gc_loop(cast(Any, object()), lambda p: None))
    await asyncio.wait_for(done.wait(), 5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _stalled_calls(error: Mock) -> list[Any]:
    return [c for c in error.call_args_list if c[0][0] == "sandbox.gc_stalled"]


async def test_gc_stalled_not_emitted_on_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = SandboxRegistry(backend=cast(Any, FakeBackend()))
    health = _fresh_health(registry)
    error = Mock()
    monkeypatch.setattr("aios.sandbox.registry.log.error", error)
    monkeypatch.setattr("aios.sandbox.registry.log.warning", Mock())

    await _drive(registry, [True, False], monkeypatch)

    assert _stalled_calls(error) == []
    assert health.consecutive_failures == 1
    assert health.stalled is False


async def test_gc_stalled_emitted_on_second_consecutive_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SandboxRegistry(backend=cast(Any, FakeBackend()))
    health = _fresh_health(registry)
    error = Mock()
    monkeypatch.setattr("aios.sandbox.registry.log.error", error)
    monkeypatch.setattr("aios.sandbox.registry.log.warning", Mock())

    await _drive(registry, [True, False, False], monkeypatch)

    stalled = _stalled_calls(error)
    assert len(stalled) == 1
    kwargs = stalled[0][1]
    assert kwargs["consecutive_failures"] == 2
    assert kwargs["error_class"] == "TimeoutError"
    assert health.last_success_at is not None
    assert kwargs["last_success_at"] == health.last_success_at.isoformat()
    assert health.snapshot()["gc_consecutive_failures"] == 2
    assert health.stalled is True


async def test_recovery_clears_published_failure_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SandboxRegistry(backend=cast(Any, FakeBackend()))
    health = _fresh_health(registry)
    monkeypatch.setattr("aios.sandbox.registry.log.error", Mock())
    monkeypatch.setattr("aios.sandbox.registry.log.warning", Mock())

    await _drive(registry, [False, False, True], monkeypatch)

    assert health.consecutive_failures == 0
    assert health.stalled is False
    assert health.last_success_at is not None
    assert health.snapshot()["gc_consecutive_failures"] == 0


async def test_enumeration_timeout_fails_the_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed-out managed-image enumeration aborts the tick and is counted."""
    from aios.sandbox.backends.base import SandboxBackendError

    backend = FakeBackend()
    registry = SandboxRegistry(backend=cast(Any, backend))
    health = _fresh_health(registry)
    monkeypatch.setattr("aios.sandbox.registry.log.warning", Mock())

    async def no_containers(**kwargs: Any) -> list[Any]:
        return []

    async def timed_out(**kwargs: Any) -> list[Any]:
        raise SandboxBackendError("docker cli timed out after 1.0s: docker images")

    monkeypatch.setattr(backend, "list_managed", no_containers, raising=False)
    monkeypatch.setattr(backend, "list_managed_images", timed_out, raising=False)
    monkeypatch.setattr(registry, "_load_gc_states", AsyncMock(return_value={}))
    monkeypatch.setattr(registry, "_gc_corpse_pass", AsyncMock())

    with pytest.raises(SandboxBackendError, match="timed out"):
        await registry._gc_once(cast(Any, object()))
    assert health.consecutive_failures == 0  # _gc_once alone does not publish


async def test_pressure_callback_failure_is_not_a_gc_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing pressure consumer must not mark a succeeding GC as stalled."""
    registry = SandboxRegistry(backend=cast(Any, FakeBackend()))
    health = _fresh_health(registry)
    error = Mock()
    exception = Mock()
    monkeypatch.setattr("aios.sandbox.registry.log.error", error)
    monkeypatch.setattr("aios.sandbox.registry.log.warning", Mock())
    monkeypatch.setattr("aios.sandbox.registry.log.exception", exception)
    monkeypatch.setattr("aios.sandbox.registry._GC_INTERVAL_SECONDS", 0)

    async def ok(pool: Any) -> GcPressureResult:
        return GcPressureResult()

    calls = 0

    def bad_callback(pressure: GcPressureResult) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("consumer broke")

    monkeypatch.setattr(registry, "_gc_once", ok)
    task = asyncio.create_task(registry._gc_loop(cast(Any, object()), bad_callback))
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert calls >= 3
    assert health.consecutive_failures == 0
    assert health.stalled is False
    assert health.last_success_at is not None
    assert _stalled_calls(error) == []
    events = [c[0][0] for c in exception.call_args_list]
    assert "sandbox.gc_pressure_callback_failed" in events
    assert "sandbox.gc_tick_failed" not in events
