"""Unit coverage for :meth:`SandboxRegistry._reap_idle_loop`.

The idle-TTL reaper is one long-lived background task that releases
sandboxes idle past ``idle_timeout``. ``release()`` calls
``backend.destroy(handle)`` which talks to Docker — any daemon hiccup
(network blip, container already gone, paused daemon) raises
``SandboxBackendError``. Pre-fix, that exception escaped the
``while True`` loop and the reaper task ended; ``start_reaper`` is
single-shot, so no idle sandbox was ever reaped again for the worker's
lifetime. Containers accumulated silently until process exit.

Same anti-pattern as ``_run_interrupt_listener`` (fixed in PR #443).
``_periodic_sweep`` (``worker.py:343``) demonstrates the correct
template: nest try/except INSIDE the ``while True``.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aios.sandbox.backends.base import SandboxBackend, SandboxBackendError, SandboxHandle
from aios.sandbox.registry import SandboxRegistry


def _handle(session_id: str) -> SandboxHandle:
    return SandboxHandle(
        sandbox_id=f"sb_{session_id}",
        session_id=session_id,
        workspace_path=Path(f"/tmp/{session_id}"),
    )


def _backend_with_destroy(destroy: Any) -> SandboxBackend:
    backend = MagicMock(spec=SandboxBackend)
    backend.destroy = destroy
    backend.name = "stub"
    return backend


async def test_reaper_survives_release_failure() -> None:
    """One ``backend.destroy`` failure must not disable the reaper.

    Two idle sandboxes; first ``release()`` propagates ``SandboxBackendError``
    from the backend; the second must still be reaped on a subsequent tick.
    """
    destroy_calls: list[SandboxHandle] = []
    both_attempted = asyncio.Event()

    async def destroy(handle: SandboxHandle) -> None:
        destroy_calls.append(handle)
        if len(destroy_calls) >= 2:
            both_attempted.set()
        if len(destroy_calls) == 1:
            raise SandboxBackendError("simulated docker hiccup")

    registry = SandboxRegistry(_backend_with_destroy(AsyncMock(side_effect=destroy)))

    # Pre-populate two idle sandboxes (last_used=0 → far past any idle_timeout).
    registry._handles["sess_a"] = _handle("sess_a")
    registry._handles["sess_b"] = _handle("sess_b")
    registry._last_used["sess_a"] = 0.0
    registry._last_used["sess_b"] = 0.0

    # Tight loop so the test completes quickly — idle_timeout=0 means
    # every entry is idle; interval=0.01 means a fresh tick every ~10ms.
    reaper = asyncio.create_task(registry._reap_idle_loop(idle_timeout=0.0, interval=0.01))
    try:
        try:
            await asyncio.wait_for(both_attempted.wait(), timeout=1.0)
        except TimeoutError as exc:
            raise AssertionError(
                f"reaper did not attempt the second release within 1s "
                f"(destroy_calls={len(destroy_calls)}, "
                f"reaper.done={reaper.done()}, "
                f"exception={reaper.exception() if reaper.done() else None}); "
                f"the first destroy's exception escaped while True and killed the task"
            ) from exc

        assert not reaper.done(), (
            f"reaper task died after one destroy failure (exception={reaper.exception()})"
        )
    finally:
        reaper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper


async def test_reaper_processes_sandboxes_added_after_initial_failure() -> None:
    """A sandbox added AFTER the initial failure is still reaped.

    Pins the survivability contract one step further: after the reaper
    survives one bad tick, future ticks pick up newly-idle entries.
    """
    destroy_calls: list[SandboxHandle] = []
    initial_attempted = asyncio.Event()
    later_attempted = asyncio.Event()

    async def destroy(handle: SandboxHandle) -> None:
        destroy_calls.append(handle)
        if handle.session_id == "sess_initial":
            initial_attempted.set()
            raise SandboxBackendError("simulated transient")
        if handle.session_id == "sess_later":
            later_attempted.set()

    registry = SandboxRegistry(_backend_with_destroy(AsyncMock(side_effect=destroy)))

    registry._handles["sess_initial"] = _handle("sess_initial")
    registry._last_used["sess_initial"] = 0.0

    reaper = asyncio.create_task(registry._reap_idle_loop(idle_timeout=0.0, interval=0.01))
    try:
        await asyncio.wait_for(initial_attempted.wait(), timeout=1.0)

        # Add a fresh idle sandbox after the first failure.
        registry._handles["sess_later"] = _handle("sess_later")
        registry._last_used["sess_later"] = 0.0

        try:
            await asyncio.wait_for(later_attempted.wait(), timeout=1.0)
        except TimeoutError as exc:
            raise AssertionError(
                f"sess_later not reaped after initial failure "
                f"(reaper.done={reaper.done()}, "
                f"exception={reaper.exception() if reaper.done() else None})"
            ) from exc
    finally:
        reaper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper
