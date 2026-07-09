"""Unit coverage for :func:`aios.harness.worker._run_interrupt_listener`.

The session-interrupt listener is one long-lived background task in the
worker. Any exception that escapes its inner dispatch — a transient
``cancel_step`` failure, a logger blip, a registry race — used to land
in the outer ``except Exception`` wrapping the ``while True``, killing
the task for the worker's lifetime. Subsequent ``POST
/v1/sessions/{id}/interrupt`` requests succeed at the API (the NOTIFY
fires) but nothing on the worker side observes them, silently disabling
a documented endpoint.

The companion ``_periodic_sweep`` next door (``worker.py:334``) nests
the ``try`` INSIDE the ``while True`` for exactly this reason. This
test pins the listener to the same survivability contract.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.worker import _run_interrupt_listener


async def test_listener_survives_dispatch_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One exception in dispatch must not disable the listener.

    Drive a real ``asyncio.Queue`` through the listener, with a
    ``InflightToolRegistry`` whose ``cancel_step`` raises on the first call.
    The listener must log the failure and continue processing
    subsequent interrupts — today the first exception escapes the
    outer try/except wrapping ``while True`` and kills the task.
    """
    queue: asyncio.Queue[str] = asyncio.Queue()

    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        yield queue

    monkeypatch.setattr(
        "aios.harness.worker.listen_for_session_interrupts",
        fake_listen,
    )

    registry = MagicMock(spec=InflightToolRegistry)
    registry.tracked_session_ids.return_value = set()
    pool = MagicMock()

    first_dispatched = asyncio.Event()
    second_dispatched = asyncio.Event()
    call_count = 0

    def cancel_step(_session_id: str) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_dispatched.set()
            raise RuntimeError("simulated transient cancel_step failure")
        second_dispatched.set()
        return True

    registry.cancel_step.side_effect = cancel_step
    registry.cancel_session.return_value = 0

    listener_task = asyncio.create_task(
        _run_interrupt_listener("postgresql://stub", registry, pool)
    )

    try:
        # First interrupt — drives cancel_step into the exception path.
        await queue.put("session_first")
        await asyncio.wait_for(first_dispatched.wait(), timeout=1.0)

        # Second interrupt — the listener must still be alive to process it.
        # If the outer-try/except bug were still in effect, the listener
        # task would have exited cleanly and ``second_dispatched`` would
        # never fire; ``wait_for`` would raise ``TimeoutError``.
        await queue.put("session_second")
        try:
            await asyncio.wait_for(second_dispatched.wait(), timeout=1.0)
        except TimeoutError as exc:
            raise AssertionError(
                f"listener did not process second interrupt within 1s — "
                f"done={listener_task.done()}, "
                f"exception="
                f"{listener_task.exception() if listener_task.done() else None}"
            ) from exc

        assert registry.cancel_step.call_count == 2
    finally:
        listener_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener_task


async def test_listener_reconnects_after_context_acquisition_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_enter_attempted = asyncio.Event()
    second_entered = asyncio.Event()
    queue: asyncio.Queue[str] = asyncio.Queue()
    attempts = 0

    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_enter_attempted.set()
            raise RuntimeError("listener connect failed")
        second_entered.set()
        yield queue

    monkeypatch.setattr(
        "aios.harness.worker.listen_for_session_interrupts",
        fake_listen,
    )
    monkeypatch.setattr("aios.harness.worker._LISTEN_RECONNECT_BACKOFF_SECONDS", 0)

    registry = MagicMock(spec=InflightToolRegistry)
    registry.tracked_session_ids.return_value = set()
    pool = MagicMock()
    listener_task = asyncio.create_task(
        _run_interrupt_listener("postgresql://stub", registry, pool)
    )
    try:
        await asyncio.wait_for(first_enter_attempted.wait(), timeout=1.0)
        await asyncio.wait_for(second_entered.wait(), timeout=1.0)
        assert attempts == 2
    finally:
        listener_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener_task


async def test_listener_cancelled_error_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        class CancelOnEnter:
            def __await__(self) -> Generator[None]:
                yield
                raise asyncio.CancelledError

        await CancelOnEnter()
        yield asyncio.Queue()

    monkeypatch.setattr(
        "aios.harness.worker.listen_for_session_interrupts",
        fake_listen,
    )
    monkeypatch.setattr("aios.harness.worker._LISTEN_RECONNECT_BACKOFF_SECONDS", 0)

    registry = MagicMock(spec=InflightToolRegistry)
    registry.tracked_session_ids.return_value = set()
    pool = MagicMock()
    with pytest.raises(asyncio.CancelledError):
        await _run_interrupt_listener("postgresql://stub", registry, pool)
