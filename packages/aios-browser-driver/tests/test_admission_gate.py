"""The admission gate: synchronous admit, drain-to-zero, timeout, reopen."""

from __future__ import annotations

import asyncio

from aios_browser_driver.takeover.state import AdmissionGate


async def test_admit_release_and_idle() -> None:
    gate = AdmissionGate()
    assert gate.admit() is True
    assert gate.admit() is True  # two in flight
    gate.release()
    gate.release()
    # Idle again → a fresh close drains instantly.
    assert await gate.close_and_drain(0.01) is True


async def test_closed_gate_refuses_admission() -> None:
    gate = AdmissionGate()
    assert await gate.close_and_drain(0.01) is True
    assert gate.closed is True
    assert gate.admit() is False


async def test_close_waits_for_in_flight_then_drains() -> None:
    gate = AdmissionGate()
    assert gate.admit() is True  # one action in flight
    drain = asyncio.create_task(gate.close_and_drain(1.0))
    await asyncio.sleep(0.01)
    assert not drain.done()  # blocked on the in-flight action
    assert gate.admit() is False  # closed the instant close_and_drain began
    gate.release()
    assert await drain is True


async def test_drain_timeout_returns_false() -> None:
    gate = AdmissionGate()
    assert gate.admit() is True
    assert await gate.close_and_drain(0.02) is False  # never released
    # The caller reopens and the gate admits again.
    gate.reopen()
    assert gate.admit() is True


async def test_admit_is_synchronous_no_await_window() -> None:
    # admit() must not await between the closed-check and the increment, or a
    # concurrent close could slip a drain in. Exercised by racing many admits
    # against a close on one event-loop tick.
    gate = AdmissionGate()
    admitted = [gate.admit() for _ in range(5)]
    assert all(admitted)
    drain = asyncio.create_task(gate.close_and_drain(1.0))
    await asyncio.sleep(0)
    assert gate.admit() is False
    for _ in range(5):
        gate.release()
    assert await drain is True
