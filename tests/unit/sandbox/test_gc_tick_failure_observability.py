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
from unittest.mock import Mock

import pytest

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
    with pytest.raises(RuntimeError, match="snapshot capacity pressure"):
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
