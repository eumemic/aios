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

Additional coverage: the reaper must hold the per-session lock while
calling ``release()``. Without the lock, a concurrent fast-path caller
can grab a handle that the reaper is about to destroy (issue #566).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aios.sandbox.backends.base import (
    SandboxBackend,
    SandboxBackendError,
    SandboxHandle,
    SnapshotOutcome,
)
from aios.sandbox.registry import SandboxRegistry


def _handle(session_id: str) -> SandboxHandle:
    return SandboxHandle(
        sandbox_id=f"sb_{session_id}",
        owner_id=session_id,
        workspace_path=Path(f"/tmp/{session_id}"),
    )


def _backend_with_destroy(destroy: Any) -> SandboxBackend:
    backend = MagicMock(spec=SandboxBackend)
    backend.destroy = destroy
    backend.name = "stub"
    # Durable session sandboxes: ``release()`` snapshots before destroying.
    # These reaper tests are about loop resilience + locking, not snapshot
    # content, so the snapshot is a no-write ``skipped_empty`` (image_id=None)
    # — that path writes NO DB pointer, so the reaper needs no ``runtime.pool``
    # and still proceeds to the ``destroy`` the tests assert on.
    backend.snapshot = AsyncMock(
        return_value=SnapshotOutcome(kind="skipped_empty", image_id=None, unique_bytes=0, depth=0)
    )
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
        if handle.owner_id == "sess_initial":
            initial_attempted.set()
            raise SandboxBackendError("simulated transient")
        if handle.owner_id == "sess_later":
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


async def test_idle_release_evicts_cache_before_destroy() -> None:
    """``_handles`` must be cleared before ``backend.destroy`` is awaited.

    ``release()`` pops from ``_handles`` synchronously before calling
    ``backend.destroy``, so any concurrent caller that checks the cache
    after the pop will see a cache miss and provision fresh — even
    though the destroy hasn't completed yet.

    This documents the ordering invariant that makes the per-session
    lock in the reaper sufficient to prevent issue #566.
    """
    handles_at_destroy: dict[str, bool] = {}
    destroy_done = asyncio.Event()

    async def destroy(handle: SandboxHandle) -> None:
        # Record whether the session is still in _handles at this point.
        handles_at_destroy[handle.owner_id] = handle.owner_id in registry._handles
        destroy_done.set()

    registry = SandboxRegistry(_backend_with_destroy(AsyncMock(side_effect=destroy)))
    registry._handles["sess_x"] = _handle("sess_x")
    registry._last_used["sess_x"] = 0.0

    reaper = asyncio.create_task(registry._reap_idle_loop(idle_timeout=0.0, interval=0.01))
    try:
        await asyncio.wait_for(destroy_done.wait(), timeout=1.0)
    finally:
        reaper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper

    # The handle must have been evicted from the cache BEFORE destroy ran.
    assert handles_at_destroy.get("sess_x") is False, (
        "release() called backend.destroy before popping _handles; "
        "a concurrent get_or_provision could return a stale handle"
    )


async def test_idle_release_holds_per_session_lock() -> None:
    """The reaper must hold the per-session lock for the full ``release()`` call.

    Without ``async with self._lock_for(sid)`` in ``_reap_idle_loop``,
    the lock is NOT held during ``backend.destroy`` — the per-session
    lock object will report ``locked() == False`` while destroy runs.

    With the fix, the reaper acquires the lock before calling
    ``release()``, so the lock is held through the entire call including
    the async ``backend.destroy`` step.

    We capture a reference to the lock object BEFORE the reaper starts
    so that ``_locks.pop()`` inside ``release()`` doesn't matter —
    Python keeps the object alive as long as we hold a reference.
    """
    destroy_started = asyncio.Event()
    destroy_continue = asyncio.Event()
    lock_state_during_destroy: dict[str, bool] = {}

    async def destroy(handle: SandboxHandle) -> None:
        lock_state_during_destroy["locked"] = captured_lock.locked()
        destroy_started.set()
        await destroy_continue.wait()

    registry = SandboxRegistry(_backend_with_destroy(AsyncMock(side_effect=destroy)))
    registry._handles["sess_x"] = _handle("sess_x")
    registry._last_used["sess_x"] = 0.0

    # Capture the lock object BEFORE the reaper touches it.
    # _lock_for() creates and stores it in _locks; the reaper will find
    # the same object via _lock_for() and acquire it.
    captured_lock = registry._lock_for("sess_x")

    reaper = asyncio.create_task(registry._reap_idle_loop(idle_timeout=0.0, interval=0.01))
    try:
        await asyncio.wait_for(destroy_started.wait(), timeout=1.0)

        assert lock_state_during_destroy.get("locked") is True, (
            "per-session lock was NOT held by the reaper during backend.destroy; "
            "fix _reap_idle_loop to wrap release() with "
            "async with self._lock_for(sid)"
        )
    finally:
        destroy_continue.set()
        reaper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper


async def test_reaper_skips_session_freshened_during_destroy_of_another() -> None:
    """The reaper must not release a session whose ``_last_used`` was
    freshened by a concurrent ``get_or_provision`` between the scan and
    the actual release.

    Scenario: two idle sessions ``sess_a`` and ``sess_b`` land in
    ``to_release`` during the scan. While the reaper is awaiting
    ``backend.destroy(handle_a)``, a tool task warm-hits
    ``get_or_provision("sess_b")``, bumping ``_last_used["sess_b"]`` to
    "now". The reaper must NOT proceed to destroy ``handle_b`` — the
    session is no longer idle.

    Pre-fix, the reaper acted on the stale ``to_release`` snapshot and
    skipped a re-check of ``_last_used`` under the per-session lock, so
    it destroyed a sandbox still in active use. The #566 lock fix did
    not cover this residual race: ``get_or_provision``'s warm path
    takes no lock, so it can freshen ``_last_used`` between the scan
    and the release without ever serializing against the reaper.
    """
    destroy_a_started = asyncio.Event()
    destroy_a_continue = asyncio.Event()
    destroy_b_calls: list[SandboxHandle] = []

    async def destroy(handle: SandboxHandle) -> None:
        if handle.owner_id == "sess_a":
            destroy_a_started.set()
            await destroy_a_continue.wait()
        elif handle.owner_id == "sess_b":
            destroy_b_calls.append(handle)

    registry = SandboxRegistry(_backend_with_destroy(AsyncMock(side_effect=destroy)))
    # Insertion order is significant: _reap_idle_once iterates
    # _last_used in dict-insertion order, so sess_a is processed first.
    handle_a = _handle("sess_a")
    handle_b = _handle("sess_b")
    registry._handles["sess_a"] = handle_a
    registry._handles["sess_b"] = handle_b
    # Both sessions are deeply idle: _last_used set 1000s in the past,
    # idle_timeout=300s — both land in to_release on the scan.
    long_ago = time.monotonic() - 1000.0
    registry._last_used["sess_a"] = long_ago
    registry._last_used["sess_b"] = long_ago

    reap_task = asyncio.create_task(registry._reap_idle_once(idle_timeout=300.0))
    try:
        await asyncio.wait_for(destroy_a_started.wait(), timeout=1.0)
        # Reaper is now parked inside backend.destroy(handle_a). Warm-hit
        # sess_b — _last_used["sess_b"] is bumped to "now", so sess_b is
        # no longer idle by the time the reaper processes it.
        returned = await registry.get_or_provision("sess_b")
        assert returned is handle_b
        destroy_a_continue.set()
        await asyncio.wait_for(reap_task, timeout=1.0)
    finally:
        if not reap_task.done():
            reap_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reap_task

    assert destroy_b_calls == [], (
        "reaper destroyed sess_b's container despite a concurrent "
        'get_or_provision freshening _last_used["sess_b"]. The reaper '
        "must re-check idleness under the per-session lock before "
        "calling release()."
    )
