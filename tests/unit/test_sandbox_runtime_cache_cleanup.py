"""Unit coverage: ``SandboxRegistry.release()`` clears per-session runtime caches.

``aios.harness.runtime`` holds two worker-process globals keyed on
``session_id`` — ``_session_memory_mounts`` and ``_session_read_shas`` —
that are populated at the top of every step (``loop.py:87``) and read by
tool calls. They were defined with paired ``clear_session_*`` helpers
documented as "after session unload", but production NEVER called them
(prior audit confirmed via repo-wide grep). Every session that ran a
step left an entry that persisted for the worker's process lifetime,
so the dicts grew unboundedly across long-running workers handling many
sessions.

The natural unload event is ``SandboxRegistry.release()``: the idle
reaper triggers it on per-session TTL, ``release_if_mounts_changed``
triggers it on mount drift, and ``release_all`` triggers it at worker
shutdown. After this PR, ``release()`` also clears the runtime caches
keyed on the released session — re-populated naturally by the next
step if the session wakes back up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness import runtime
from aios.models.memory_stores import MemoryStoreResourceEcho
from aios.sandbox.backends.base import SandboxBackend, SandboxHandle
from aios.sandbox.registry import SandboxRegistry


def _handle(session_id: str) -> SandboxHandle:
    return SandboxHandle(
        sandbox_id=f"sb_{session_id}",
        session_id=session_id,
        workspace_path=Path(f"/tmp/{session_id}"),
    )


def _backend() -> SandboxBackend:
    backend = MagicMock(spec=SandboxBackend)
    backend.destroy = AsyncMock(return_value=None)
    backend.name = "stub"
    return backend


def _echo(store_id: str, name: str) -> MemoryStoreResourceEcho:
    return MemoryStoreResourceEcho(
        memory_store_id=store_id,
        access="read_write",
        instructions="",
        name=name,
        description="",
        mount_path=f"/mnt/memory/{name}",
    )


@pytest.fixture(autouse=True)
def _reset_runtime_caches() -> Any:
    """Clear the module-level caches between tests so leak from one
    test can't make another's assertion incidentally pass."""
    runtime._session_memory_mounts.clear()
    runtime._session_read_shas.clear()
    yield
    runtime._session_memory_mounts.clear()
    runtime._session_read_shas.clear()


async def test_release_clears_session_memory_mounts() -> None:
    """A session's ``runtime._session_memory_mounts`` entry must be gone
    after ``SandboxRegistry.release(session_id)`` runs."""
    registry = SandboxRegistry(_backend())
    sid = "sess_mount_test"
    registry._handles[sid] = _handle(sid)

    echo = _echo("ms_x", "store-x")
    runtime.set_session_memory_mounts(sid, [echo])
    assert runtime.get_session_memory_mounts(sid) == [echo]

    await registry.release(sid)

    assert runtime.get_session_memory_mounts(sid) == [], (
        "release() should have cleared the session's memory-mounts cache; "
        "the entry persists across the worker's lifetime → unbounded growth"
    )


async def test_release_clears_session_read_shas() -> None:
    """A session's ``runtime._session_read_shas`` entry must be gone
    after ``SandboxRegistry.release(session_id)`` runs."""
    registry = SandboxRegistry(_backend())
    sid = "sess_shas_test"
    registry._handles[sid] = _handle(sid)

    runtime.set_read_sha(sid, "store_a", "/path/a", "sha_a")
    assert runtime.get_read_sha(sid, "store_a", "/path/a") == "sha_a"

    await registry.release(sid)

    assert runtime.get_read_sha(sid, "store_a", "/path/a") is None, (
        "release() should have cleared the session's read-sha cache; "
        "the entry persists across the worker's lifetime → unbounded growth"
    )


async def test_release_does_not_disturb_other_sessions_caches() -> None:
    """Releasing session A must not affect session B's cache entries."""
    registry = SandboxRegistry(_backend())
    registry._handles["sess_a"] = _handle("sess_a")
    registry._handles["sess_b"] = _handle("sess_b")

    echo_a = _echo("ms_a", "store-a")
    echo_b = _echo("ms_b", "store-b")
    runtime.set_session_memory_mounts("sess_a", [echo_a])
    runtime.set_session_memory_mounts("sess_b", [echo_b])
    runtime.set_read_sha("sess_a", "store", "/p", "sha_a")
    runtime.set_read_sha("sess_b", "store", "/p", "sha_b")

    await registry.release("sess_a")

    assert runtime.get_session_memory_mounts("sess_a") == []
    assert runtime.get_session_memory_mounts("sess_b") == [echo_b]
    assert runtime.get_read_sha("sess_a", "store", "/p") is None
    assert runtime.get_read_sha("sess_b", "store", "/p") == "sha_b"
