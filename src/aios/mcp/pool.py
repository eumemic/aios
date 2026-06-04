"""Worker-scoped MCP session pool — per-call session checkout.

Holds initialized ``ClientSession`` instances per ``(url, vault_id, headers_key)``
key (``headers_key`` hashes only the static spec headers, so the key is stable
across OAuth token rotation — see :meth:`acquire` and #459) so tool
discovery and invocation can reuse an already-initialized MCP connection
instead of opening a fresh one on every call.

Checkout model: each tool call :meth:`acquire`\\ s an entry for **exclusive**
use and :meth:`release`\\ s (healthy) or :meth:`discard`\\ s (broken) it when
done. Because an entry — and its :class:`HttpErrorSink` — is never shared
while checked out, two concurrent calls to the same server can't cross-attribute
each other's HTTP errors (the bug a single shared session caused). Released
entries return to a per-key idle pool and are reused on the next acquire (warm
reuse); a fresh session is opened only when none are idle.

Pool lifecycle:

- Created at worker startup, stashed on :mod:`aios.harness.runtime`.
- :meth:`acquire` pops an idle entry or opens one. At most
  ``MAX_SESSIONS_PER_KEY`` live sessions exist per key — beyond the cap the
  caller waits on a per-key :class:`asyncio.Condition` until a release/discard
  frees a slot. The Condition is held across the open so the cap is strict.
- :meth:`release` returns a healthy session to idle; :meth:`discard` drops a
  broken one (fire-and-forget close). A benign HTTP error (e.g. 403) releases —
  the transport is fine; only transport failures discard.
- The idle reaper closes sessions idle longer than the TTL (idle entries only —
  an in-use entry is by definition active).
- :meth:`close_all` is called from ``worker_main``'s ``finally`` at shutdown and
  closes **both** idle and in-use sessions (in-use owner tasks must receive the
  shutdown signal or they leak).

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

# Cap on live sessions per ``(url, vault_id, headers_key)``. Bounds session/connection growth
# under a model that fires many concurrent calls at one server; at the cap an
# ``acquire`` waits for a release rather than opening unboundedly. A session is
# created only when in-use < cap and none are idle, so the total live count never
# exceeds the peak concurrent in-use count, which is bounded by this value.
MAX_SESSIONS_PER_KEY = 8

type _PoolKey = tuple[str, str | None, str]  # (url, vault_id, headers_key)


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
        # monotonic() stamped each time this entry is released back to idle.
        # The reaper keys off this, NOT owner-task liveness: the task parks on
        # `await shutdown.wait()` for the entry's whole life, so liveness is a
        # useless staleness signal. Only idle entries are reaped (an in-use
        # entry is by definition active), so last_used is the time the entry
        # last finished a checkout.
        #
        # Post-#459 the pool keys on (url, vault_id, headers_key), stable
        # across OAuth refresh (headers_key hashes only the static spec
        # headers). The reaper's remaining job is the cold-entry
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
    """Worker-scoped pool of MCP ``ClientSession`` instances, checked out per call.

    Single-event-loop — no thread-safety concerns. A per-key ``asyncio.Condition``
    guards the idle/in-use sets and blocks acquires at the per-key cap.
    """

    def __init__(self) -> None:
        # Idle (available) and in-use (checked out) entries per key. The cap is
        # on in-use count; idle entries are reused before a new one is opened.
        self._idle: dict[_PoolKey, list[_Entry]] = {}
        self._in_use: dict[_PoolKey, set[_Entry]] = {}
        self._conditions: dict[_PoolKey, asyncio.Condition] = {}
        self._closed = False
        self._reaper_task: asyncio.Task[None] | None = None
        # Strong refs for discard()'s fire-and-forget close tasks. asyncio
        # only weak-refs tasks, so without this the task can be GC'd
        # before close() unwinds the owner task's contexts — defeating
        # the leak fix.
        self._close_tasks: set[asyncio.Task[None]] = set()

    def _condition_for(self, key: _PoolKey) -> asyncio.Condition:
        cond = self._conditions.get(key)
        if cond is None:
            cond = asyncio.Condition()
            self._conditions[key] = cond
        return cond

    def _drop_in_use(self, key: _PoolKey, entry: _Entry) -> None:
        """Remove ``entry`` from the in-use set, deleting the set once it empties.

        Mirrors acquire's idle-list cleanup so a key whose last in-flight entry
        is released/discarded doesn't leave a ghost empty set behind. Caller
        holds the key's Condition.
        """
        in_use = self._in_use.get(key)
        if in_use is None:
            return
        in_use.discard(entry)
        if not in_use:
            del self._in_use[key]

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

    async def acquire(
        self, url: str, vault_id: str | None, headers_key: str, headers: dict[str, str]
    ) -> _Entry:
        """Check out an entry for **exclusive** use; caller must release/discard it.

        Keyed on ``(url, vault_id, headers_key)``. ``vault_id`` keeps the key
        stable across OAuth token rotation because it is the row id of the
        ``vault_credentials`` entry that ``refresh_credential`` updates in
        place (see #459). ``headers_key`` hashes ONLY the spec's static
        config headers (never the merged auth headers), so token rotation
        does not change it either — preserving #459. ``vault_id`` is ``None``
        for the no-credential case (an unauthenticated MCP server); all such
        callers on the same URL with the same static headers share one key,
        which is safe because no auth identity is keyed on.

        Reuses an idle entry if one exists; otherwise opens a fresh one when
        under the per-key cap; otherwise waits on the per-key Condition for a
        release/discard to free a slot. The Condition is held across the open so
        the cap is strict (no thundering-herd over-open). An ``_open_entry``
        failure propagates out, releasing the Condition; no slot is consumed and
        no waiter is notified — the next waiter to wake simply retries.
        """
        if self._closed:
            raise RuntimeError("McpSessionPool is closed")
        key: _PoolKey = (url, vault_id, headers_key)
        cond = self._condition_for(key)
        async with cond:
            while True:
                idle = self._idle.get(key)
                if idle:
                    entry = idle.pop()
                    if not idle:
                        del self._idle[key]
                    self._in_use.setdefault(key, set()).add(entry)
                    return entry
                if len(self._in_use.get(key, set())) < MAX_SESSIONS_PER_KEY:
                    log.info("mcp_pool.connecting", url=url)
                    entry = await self._open_entry(url, headers)
                    self._in_use.setdefault(key, set()).add(entry)
                    log.info("mcp_pool.connected", url=url)
                    return entry
                await cond.wait()

    async def release(
        self, url: str, vault_id: str | None, headers_key: str, entry: _Entry
    ) -> None:
        """Return a healthy entry to the idle pool for reuse.

        Called when the session is fine — a successful call, or a benign
        application-level HTTP error (the transport is unaffected). Stamps
        ``last_used`` (the reaper's staleness signal) and notifies one waiter
        that a slot is free.
        """
        entry.last_used = time.monotonic()
        key: _PoolKey = (url, vault_id, headers_key)
        cond = self._condition_for(key)
        async with cond:
            self._drop_in_use(key, entry)
            self._idle.setdefault(key, []).append(entry)
            cond.notify()

    async def discard(
        self, url: str, vault_id: str | None, headers_key: str, entry: _Entry
    ) -> None:
        """Drop a broken entry and close it in the background.

        Called when the session may be broken (transport failure, cancellation).
        The caller can't block on close (the owner task may wait up to the httpx
        read timeout before unwinding), so the close is fire-and-forget. Skipping
        close entirely would strand the owner task + httpx client + SSE stream —
        the same leak the idle reaper exists to prevent. Frees the in-use slot
        and notifies one waiter.
        """
        key: _PoolKey = (url, vault_id, headers_key)
        cond = self._condition_for(key)
        async with cond:
            self._drop_in_use(key, entry)
            cond.notify()
        task = asyncio.create_task(entry.close(), name=f"mcp-pool-discard:{url}")
        self._close_tasks.add(task)
        task.add_done_callback(self._close_tasks.discard)
        log.info("mcp_pool.discarded", url=url)

    async def _reap_idle_once(self, *, idle_timeout: float, now: float) -> None:
        """Close idle entries unused longer than ``idle_timeout`` seconds.

        Idle entries only — an in-use (checked-out) entry is by definition
        active. Keyed on ``_Entry.last_used`` (stamped at :meth:`release`).
        Reclamation goes through ``_Entry.close()`` (same path as
        :meth:`close_all`), NOT a bare ``remove``: dropping the reference alone
        strands the owner task, httpx client and SSE stream — exactly the leak
        this reaps. No notify after close: reaping an idle entry doesn't free an
        in-use slot, so no capped waiter is unblocked by it.
        """
        if self._closed:
            return
        stale = [
            (key, entry)
            for key, entries in self._idle.items()
            for entry in entries
            if now - entry.last_used > idle_timeout
        ]
        for key, entry in stale:
            async with self._condition_for(key):
                # Re-check under the lock: acquire() may have popped this entry
                # (warm reuse) between the scan and here, or release() may have
                # refreshed its last_used — same TOCTOU shape as the
                # SandboxRegistry reaper fixed in #654.
                idle = self._idle.get(key)
                if idle is None or entry not in idle or now - entry.last_used <= idle_timeout:
                    continue
                idle.remove(entry)
                if not idle:
                    del self._idle[key]
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
        """Tear down all sessions — idle AND in-use. Called at worker shutdown.

        In-use entries must be closed too: their owner tasks are parked on the
        shutdown event and would leak otherwise (setting it unwinds the
        AsyncExitStack, cancelling any in-flight transport task). ``_closed`` is
        set first so no new session is acquired during or after teardown.
        """
        self._closed = True
        entries = [e for entries in self._idle.values() for e in entries]
        entries += [e for entries in self._in_use.values() for e in entries]
        self._idle.clear()
        self._in_use.clear()
        self._conditions.clear()
        if not entries:
            return
        log.info("mcp_pool.close_all", count=len(entries))
        for entry in entries:
            try:
                await entry.close()
            except Exception:
                log.warning("mcp_pool.close_entry_failed", exc_info=True)
