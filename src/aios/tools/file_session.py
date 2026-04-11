"""Per-session state for the file tools.

Unlike the stateless bash tool, the file tools need continuity across
calls within a session:

- ``ShellFileOperations._command_cache`` caches ``rg`` / ``grep``
  availability checks.
- Staleness tracking warns when the model tries to edit a file that was
  modified externally since the last read.
- Consecutive-read dedup warns (and then blocks) the model from reading
  the same file range over and over.

All of that lives in a :class:`FileToolSession` keyed by ``session_id``.
The cache is pinned to the sandbox container's lifecycle: when the
sandbox registry evicts or releases a handle, :func:`evict` is called
here to drop the matching session. The next file tool call then walks
through :func:`get_or_create`, which provisions a fresh container
through the sandbox registry and binds a fresh adapter + fresh
``ShellFileOperations`` with empty tracker state.

Lifecycle invariant: a ``FileToolSession`` always references a live
``ContainerHandle`` through its ``SandboxTerminalEnv``. When the
container dies (and therefore the handle is evicted from the sandbox
registry), the ``FileToolSession`` is evicted too.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from aios.harness import runtime
from aios.logging import get_logger
from aios.tools.adapters import SandboxTerminalEnv
from aios.vendor.hermes_files.file_operations import ShellFileOperations

log = get_logger("aios.tools.file_session")


@dataclass
class FileToolSession:
    """Per-session file-tool state: cached ops object + tracker dicts.

    Fields:
        file_ops: The cached :class:`ShellFileOperations` bound to this
            session's container (via a :class:`SandboxTerminalEnv`).
        last_key: The most recently seen ``(path, offset, limit)`` key
            from a read call. Used for consecutive-read detection.
        consecutive: How many times the same ``last_key`` has been read
            in a row. Warn at 3, hard-block at 4 (see read handler).
        read_history: Set of every ``(path, offset, limit)`` key the
            model has read this session. Used to decide whether a write
            is against a "stale" file.
        dedup: Map from ``(path, offset, limit)`` to the mtime at which
            the content was last returned. A subsequent identical read
            that finds the same mtime returns a lightweight stub.
        read_timestamps: Map from path to the mtime at which it was
            last read by this session. Used to detect external
            modification between read and write.
        lock: An ``asyncio.Lock`` protecting the above mutable state.
            Held for the duration of any handler call, which serializes
            tool calls for a single session against this session's
            tracker. Cross-session concurrency is unaffected.
    """

    file_ops: ShellFileOperations
    # last_key is a heterogeneous tuple: read keys are
    # ``(path, offset, limit)``; search keys start with the literal
    # ``"search"`` and include the full search arg set. Typed as
    # ``tuple[Any, ...]`` because mixing read/search flavours in the
    # same slot is load-bearing for "read-then-search resets the
    # consecutive counter".
    last_key: tuple[Any, ...] | None = None
    consecutive: int = 0
    read_history: set[tuple[str, int, int]] = field(default_factory=set)
    dedup: dict[tuple[str, int, int], float] = field(default_factory=dict)
    read_timestamps: dict[str, float] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Module-level state. Per-worker-process, not persisted across restarts.
_sessions: dict[str, FileToolSession] = {}
_sessions_lock = asyncio.Lock()


async def get_or_create(session_id: str) -> FileToolSession:
    """Return the cached :class:`FileToolSession` for ``session_id``.

    On first call for a session (or after an eviction), this walks the
    slow path:

    1. Ensure the sandbox registry has a live container for this session
       (provisioning one lazily if needed).
    2. Wrap the container handle in a :class:`SandboxTerminalEnv`.
    3. Construct a :class:`ShellFileOperations` bound to that adapter.
    4. Wrap everything in a fresh :class:`FileToolSession` and cache it.

    Subsequent calls for the same session return the cached entry.
    """
    async with _sessions_lock:
        existing = _sessions.get(session_id)
        if existing is not None:
            return existing

        sandbox = runtime.require_sandbox_registry()
        handle = await sandbox.get_or_provision(session_id)
        env = SandboxTerminalEnv(handle)
        file_ops = ShellFileOperations(env)
        sess = FileToolSession(file_ops=file_ops)
        _sessions[session_id] = sess
        log.info("file_session.created", session_id=session_id)
        return sess


def evict(session_id: str) -> None:
    """Drop the cached :class:`FileToolSession` for ``session_id``.

    Called from :mod:`aios.sandbox.registry` when a container handle is
    evicted or released, via a lazy import (to avoid the
    ``aios.tools`` -> ``aios.sandbox`` -> ``aios.tools`` cycle).

    Synchronous so it can be called from the sandbox registry's
    synchronous ``evict()`` method without awaiting.
    """
    if _sessions.pop(session_id, None) is not None:
        log.info("file_session.evicted", session_id=session_id)
