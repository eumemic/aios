"""Worker-scoped MCP session pool.

Holds a persistent ``ClientSession`` per ``(url, vault_id)`` key (stable
across OAuth token rotation — see :func:`get_or_connect` and #459) so
tool discovery and invocation can reuse an already-initialized MCP
connection instead of opening a fresh one on every call.

Pool lifecycle:

- Created at worker startup, stashed on :mod:`aios.harness.runtime`.
- :meth:`get_or_connect` lazily opens and initializes a session on first
  use. A per-key ``asyncio.Lock`` prevents thundering-herd
  double-initialization.
- Callers evict on first failure; the next :meth:`get_or_connect`
  re-opens. Eviction drops the reference without attempting to close the
  broken session — closing a half-dead stack may hang.
- :meth:`close_all` is called from ``worker_main``'s ``finally`` at
  shutdown to tear everything down.

Owner-task model (#425): each entry's ``AsyncExitStack`` lives inside a
dedicated long-lived task. The task opens the contexts, signals the
caller they're ready, then awaits a shutdown event. ``close()`` sets the
event and the contexts exit in the same task that entered them. This
avoids the anyio "Attempted to exit cancel scope in a different task
than it was entered in" error that would otherwise fire on
:meth:`close_all` because the streamable-http client binds its cancel
scope to the opening task.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import InitializeResult

from aios.logging import get_logger

log = get_logger("aios.mcp.pool")

# Mirrors the per-call bound used by :mod:`aios.mcp.client`. The pool's
# pooled sessions share a connection-level timeout so a stalled MCP server
# can't keep the worker on a dead socket indefinitely.
_MCP_HTTPX_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

type _PoolKey = tuple[str, str | None]


@dataclass
class HttpErrorSink:
    """Captures the most recent 4xx/5xx HTTP response seen on a pooled MCP
    transport.

    The streamable-http client raises the HTTP error inside a long-lived anyio
    task group that stays suspended while the pooled session is parked, so a
    failed ``call_tool`` would otherwise hang until the tool-call timeout and
    surface a bare ``TimeoutError`` — losing the server's actual message. This
    sink lets the caller observe the error response immediately (via ``event``)
    and report it (``status`` + ``body``).
    """

    event: asyncio.Event = field(default_factory=asyncio.Event)
    status: int | None = None
    body: str = ""

    def reset(self) -> None:
        self.event.clear()
        self.status = None
        self.body = ""

    def record(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        self.event.set()


def _make_error_hook(sink: HttpErrorSink) -> Callable[[httpx.Response], Awaitable[None]]:
    """httpx response hook that records error responses into ``sink``.

    Only reads the body for 4xx/5xx (error JSON, never an SSE stream), so
    success-path streaming responses are untouched.
    """

    async def _hook(response: httpx.Response) -> None:
        if response.status_code < 400 or sink.event.is_set():
            return
        try:
            await response.aread()
            body = response.text[:800]
        except Exception:
            body = ""
        sink.record(response.status_code, body)

    return _hook


class _Entry:
    """A single pooled MCP session with its associated owner task.

    The session's contexts (httpx client, streamable-http client,
    ClientSession) all live inside ``_owner_task``. ``close()`` sets
    ``_shutdown`` so the task exits the contexts in the same task that
    entered them — see the module docstring for the anyio cancel-scope
    background.
    """

    def __init__(
        self,
        session: ClientSession,
        init_result: InitializeResult,
        shutdown: asyncio.Event,
        owner_task: asyncio.Task[None],
        last_used: float,
        error_sink: HttpErrorSink,
    ) -> None:
        self.session = session
        self.init_result = init_result
        self._shutdown = shutdown
        self._owner_task = owner_task
        # Records error responses on this session's transport so a failed
        # call_tool can fail fast with the server's message (see HttpErrorSink).
        self.error_sink = error_sink
        # monotonic() of the last get_or_connect that returned this
        # entry. Reaper keys off this, NOT owner-task liveness: the
        # task parks on `await shutdown.wait()` for the entry's whole
        # life, so liveness is a useless staleness signal.
        #
        # Post-#459 the pool keys on (url, vault_id), stable across
        # OAuth refresh. The reaper's remaining job is the cold-entry
        # vector (a vault whose tenant never returns) — defense-in-depth
        # signed off in #459's planning round, not silent accretion.
        self.last_used = last_used

    async def close(self) -> None:
        """Signal the owner task to exit its contexts and await its completion.

        Safe to call repeatedly; the owner task only sees the first set().

        The owner task's contexts may raise during exit (e.g. the remote
        MCP server already closed the socket). The caller —
        :meth:`McpSessionPool.close_all` — already swallows and logs such
        errors; we surface as if shutdown completed.
        """
        self._shutdown.set()
        with contextlib.suppress(Exception):
            await self._owner_task


class McpSessionPool:
    """Worker-scoped pool of persistent MCP ``ClientSession`` instances.

    Single-event-loop — no thread-safety concerns. Concurrent async tasks
    calling the same key serialise through a per-key ``asyncio.Lock``.
    """

    def __init__(self) -> None:
        self._entries: dict[_PoolKey, _Entry] = {}
        self._locks: dict[_PoolKey, asyncio.Lock] = {}
        self._reaper_task: asyncio.Task[None] | None = None
        # Strong refs for evict()'s fire-and-forget close tasks. asyncio
        # only weak-refs tasks, so without this the task can be GC'd
        # before close() unwinds the owner task's contexts — defeating
        # the leak fix.
        self._evict_close_tasks: set[asyncio.Task[None]] = set()

    def _lock_for(self, key: _PoolKey) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def _open_entry(self, url: str, headers: dict[str, str]) -> _Entry:
        """Open a fresh session inside a dedicated long-lived owner task.

        The contexts (httpx client, streamable-http client, ClientSession)
        all enter and exit in the same task — see the module docstring on
        the anyio cancel-scope constraint. The task signals ``ready`` once
        the session is initialized, then awaits ``shutdown`` so the
        contexts stay open for the entry's lifetime.
        """
        ready = asyncio.Event()
        shutdown = asyncio.Event()
        sink = HttpErrorSink()
        result: dict[str, Any] = {}

        async def _own() -> None:
            try:
                async with AsyncExitStack() as stack:
                    http_client: Any = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=headers,
                            timeout=_MCP_HTTPX_TIMEOUT,
                            event_hooks={"response": [_make_error_hook(sink)]},
                        )
                    )
                    read_stream, write_stream, _ = await stack.enter_async_context(
                        streamable_http_client(url, http_client=http_client)
                    )
                    session = await stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
                    result["session"] = session
                    result["init"] = await session.initialize()
                    ready.set()
                    await shutdown.wait()
            except BaseException as exc:
                result["error"] = exc
                # Unblock the caller waiting on ready so it doesn't hang
                # forever when the open path failed.
                ready.set()
                raise

        task = asyncio.create_task(_own(), name=f"mcp-pool:{url}")
        await ready.wait()
        if "error" in result:
            # The task already raised; await it to surface the original
            # exception cleanly (and clear the unhandled-exception slot).
            await task
            # Defensive — unreachable in practice.
            raise RuntimeError("mcp pool owner task signalled ready but produced no session")
        return _Entry(result["session"], result["init"], shutdown, task, time.monotonic(), sink)

    async def get_or_connect(
        self, url: str, vault_id: str | None, headers: dict[str, str]
    ) -> tuple[ClientSession, InitializeResult]:
        """Return a cached session, opening a fresh one if none exists.

        Keyed on ``(url, vault_id)`` — stable across OAuth token rotation
        because ``vault_id`` is the row id of the ``vault_credentials``
        entry that ``refresh_credential`` updates in place (see #459).
        ``vault_id`` is ``None`` for the no-credential case (an
        unauthenticated MCP server); all such callers on the same URL
        share one entry, which is safe because no auth identity is
        being keyed on.

        Double-checked locking: the fast unsynchronised check avoids
        lock contention on the common (already-connected) path; the
        slow path acquires the per-key lock and re-checks before
        opening.
        """
        key: _PoolKey = (url, vault_id)

        entry = self._entries.get(key)
        if entry is not None:
            entry.last_used = time.monotonic()
            return entry.session, entry.init_result

        async with self._lock_for(key):
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_used = time.monotonic()
                return entry.session, entry.init_result

            log.info("mcp_pool.connecting", url=url)
            entry = await self._open_entry(url, headers)
            self._entries[key] = entry
            log.info("mcp_pool.connected", url=url)

        return entry.session, entry.init_result

    def error_sink(self, url: str, vault_id: str | None) -> HttpErrorSink | None:
        """The error sink for a connected entry (or ``None`` if not connected).

        Lets the caller observe an error response on the pooled transport and
        fail fast instead of hanging to the tool-call timeout.
        """
        entry = self._entries.get((url, vault_id))
        return entry.error_sink if entry is not None else None

    def evict(self, url: str, vault_id: str | None) -> None:
        """Drop a cache entry and close it in the background.

        Called when a cached session has been found to be broken. The
        caller is on a retry path and can't block on close (the owner
        task may wait up to the httpx read timeout before unwinding), so
        the close is fire-and-forget. Skipping close entirely would
        strand the owner task + httpx client + SSE stream — same leak
        shape the idle reaper exists to prevent, except evict pops the
        entry from ``_entries`` so the reaper can't reach it.
        """
        key: _PoolKey = (url, vault_id)
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        task = asyncio.create_task(entry.close(), name=f"mcp-pool-evict:{url}")
        self._evict_close_tasks.add(task)
        task.add_done_callback(self._evict_close_tasks.discard)
        log.info("mcp_pool.evicted", url=url)

    async def _reap_idle_once(self, *, idle_timeout: float, now: float) -> None:
        """Close entries idle longer than ``idle_timeout`` seconds.

        Keyed on ``_Entry.last_used``, not owner-task liveness — see
        :class:`_Entry`. Reclamation goes through ``_Entry.close()``
        (same path as :meth:`close_all`), NOT a bare ``pop``: dropping
        the reference alone strands the owner task, httpx client and SSE
        stream — exactly the leak this reaps.
        """
        stale = [k for k, e in self._entries.items() if now - e.last_used > idle_timeout]
        for key in stale:
            async with self._lock_for(key):
                # Re-check under the lock: a concurrent get_or_connect
                # warm hit (pool.py:193-196) bumps entry.last_used
                # synchronously without acquiring the lock, so a freshen
                # can race between the scan above and this iteration.
                # Without the re-check, the reaper would close a session
                # actively held by a caller — same shape as the
                # SandboxRegistry reaper TOCTOU fixed in #654. evict()
                # is also lockless and can drop the entry between scan
                # and lock-acquire (the get() returning None handles
                # that).
                entry = self._entries.get(key)
                if entry is None or now - entry.last_used <= idle_timeout:
                    continue
                del self._entries[key]
                log.info("mcp_pool.idle_close", url=key[0])
                await entry.close()

    async def _reap_idle_loop(self, idle_timeout: float, interval: float = 60.0) -> None:
        """Background loop: close pooled sessions idle > idle_timeout.

        The try/except is nested INSIDE ``while True`` (mirroring
        :meth:`aios.sandbox.registry.SandboxRegistry._reap_idle_loop`
        and the PR #443 fix for ``_run_interrupt_listener``): a teardown
        error from one entry's ``close()`` must not silently disable the
        reaper for the worker's lifetime — that would leak every
        subsequently-superseded entry until process exit.
        ``CancelledError`` is not an :class:`Exception` subclass, so
        :meth:`stop_reaper`'s cancel still exits the loop cleanly.
        """
        while True:
            try:
                await asyncio.sleep(interval)
                await self._reap_idle_once(idle_timeout=idle_timeout, now=time.monotonic())
            except Exception:
                log.exception("mcp_pool.reap_idle_loop_failed")

    def start_reaper(self, idle_timeout: float) -> None:
        """Start the idle-TTL reaper background task."""
        if self._reaper_task is not None:
            return
        self._reaper_task = asyncio.create_task(
            self._reap_idle_loop(idle_timeout),
            name="mcp-pool-idle-reaper",
        )

    def stop_reaper(self) -> None:
        """Cancel the idle-TTL reaper."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            self._reaper_task = None

    async def close_all(self) -> None:
        """Tear down all pooled sessions. Called at worker shutdown."""
        entries = list(self._entries.values())
        self._entries.clear()
        self._locks.clear()
        if not entries:
            return
        log.info("mcp_pool.close_all", count=len(entries))
        for entry in entries:
            try:
                await entry.close()
            except Exception:
                log.warning("mcp_pool.close_entry_failed", exc_info=True)
