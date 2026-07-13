"""LISTEN/NOTIFY helper for SSE event tailing.

The Phase 2 SSE endpoint ``GET /v1/sessions/{id}/stream`` opens a long-lived
HTTP response that streams events as they're appended to the session log.
The implementation uses Postgres ``LISTEN``/``NOTIFY``: every
:func:`aios.db.queries.append_event` call issues a ``pg_notify`` after
committing the event row, and SSE clients listen on the matching channel.

This module provides the building block: an async context manager that opens
a **dedicated** asyncpg connection (NOT from the pool), runs ``LISTEN
events_<session_id>``, and yields an :class:`asyncio.Queue` that receives
event ids as they arrive.

## Why a dedicated connection

SSE responses can be very long-lived (hours). Borrowing from the pool would
either starve other consumers or, worse, leak ``LISTEN`` state across pool
borrowers. asyncpg's pool resets borrowed connections, but a still-LISTENing
connection has subtle semantics that are easier to avoid by simply opening
a fresh connection per SSE client.

## Why ``terminate()`` instead of ``await conn.close()``

The context manager's ``finally`` block must close the connection even when
the caller is being cancelled — which is exactly what happens on SSE client
disconnect, where sse-starlette cancels the response task group. Under
anyio's scope-cancellation, every ``await`` inside the cancelled scope
re-raises ``CancelledError`` immediately, so ``await conn.close()`` can't
run to completion: asyncpg's close protocol stops before transport.abort()
fires and the Postgres backend lingers until TCP keepalive reaps it
(~2h). ``conn.terminate()`` is synchronous — it calls ``transport.abort()``
directly, closing the socket without awaiting. Non-graceful but appropriate
for SSE-style disconnects where the client is already gone, and equally
correct on normal exit and exception unwinds.

## Why the callback must be synchronous

asyncpg invokes the listener callback on the connection's read loop. The
callback IS NOT itself a coroutine — calling ``await`` inside it raises.
Use ``queue.put_nowait`` and let an async consumer drain.

## Why LISTEN must precede backfill in the SSE handler

This is the most important invariant the SSE generator depends on, even
though it lives one layer up:

The SSE handler must run ``LISTEN`` BEFORE issuing the backfill ``SELECT``,
not the other way around. Otherwise events that commit during the backfill
window are silently lost — they NOTIFY before the listener is set up, the
notification has nowhere to go, and the SELECT (which only sees rows
committed before its own snapshot started) doesn't see them either.

With LISTEN-first, every event committed during the backfill window pushes
a notification into the queue. The dedup pattern is "track ``cursor`` =
max(seq) seen during backfill, skip live events with seq <= cursor". The
notification fires at commit time, which is strictly after the row is
visible to a fresh SELECT, so the listener will always see something at
least as new as the backfill cursor.

The Phase 1 :func:`aios.db.queries.append_event` issues its NOTIFY OUTSIDE
the transaction block (after commit). This is the correct ordering — a
subscriber must not see a payload for a row that isn't yet committed.
**Don't move the NOTIFY inside the transaction.** Add a fat comment if
anyone tries.

## Why ``open_listen_for_*`` exists alongside ``listen_for_*``

The four route-level SSE handlers used to instantiate ``listen_for_*`` as
an ``@asynccontextmanager`` INSIDE the generator passed to
``EventSourceResponse``.  That meant the ``asyncpg.connect`` + ``add_listener``
setup happened AFTER 200 OK headers were already on the wire — any failure
left the client holding a half-open chunked stream and the server logging
"ASGI callable returned without completing response."

``open_listen_for_*`` returns a ready-to-use :class:`ListenSubscription`,
letting the route handler preflight setup BEFORE constructing
``EventSourceResponse``.  Preflight failure surfaces as a clean 503 with
proper headers; only on success is the subscription handed to the
generator, which owns ``subscription.terminate()`` in a finally.

The ``@asynccontextmanager`` wrappers stay for callers that don't need the
two-phase shape (the long-poll ``wait_for_events`` endpoint, the
session-interrupt and connector-result listeners that resolve fast
enough that mid-body failures are negligible, plus the existing test
suite).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import asyncpg

from aios.db.pool import (
    LISTENER_TCP_KEEPALIVE_SETTINGS,
    listener_application_name,
    normalize_dsn,
)
from aios.db.sse_lock import acquire_subscriber_lock
from aios.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from contextlib import AbstractAsyncContextManager

log = get_logger("aios.db.listen")

# Process-local admission limit for route-level SSE listeners. Infrastructure
# listeners continue to call _connect_listener directly and are not constrained.
_SSE_SUBSCRIBER_LIMIT = 32
_sse_subscriber_count = 0


class SSESubscriberCapacityError(RuntimeError):
    """The process has no remaining dedicated SSE listener capacity."""


def _reserve_sse_subscriber() -> Callable[[], None]:
    """Reserve one SSE slot and return an idempotent synchronous releaser."""
    global _sse_subscriber_count
    if _sse_subscriber_count >= _SSE_SUBSCRIBER_LIMIT:
        raise SSESubscriberCapacityError(
            f"concurrent SSE subscriber limit ({_SSE_SUBSCRIBER_LIMIT}) reached"
        )
    _sse_subscriber_count += 1
    released = False

    def release() -> None:
        nonlocal released
        global _sse_subscriber_count
        if not released:
            released = True
            _sse_subscriber_count -= 1

    return release


# Sentinel payload fired on ``events_<session_id>`` when a session is archived
# mid-flight (see :func:`aios.services.sessions.archive_session`). The
# ``events_`` channel otherwise carries committed-event ids (``evt_…``) and
# transient streaming-delta JSON (``{"delta": …}``); this sentinel is neither —
# it's a content-free wake poke so consumers blocked on the channel (the
# ``await`` primitive, the long-poll ``wait`` endpoint, the SSE ``/stream``)
# re-read and observe the archive instead of sitting until their own timeout.
# It deliberately does NOT start with ``{`` so it's distinguishable from a
# delta payload, and isn't an ``evt_`` id so consumers don't try to fetch a row.
EVENTS_ARCHIVED_NOTIFY = "archived"


async def _connect_listener(db_url: str) -> asyncpg.Connection[object]:
    """Open a dedicated (non-pooled) asyncpg connection tagged with this
    instance's listener ``application_name``.

    Single injection point for every LISTEN connection in this module.
    The tag (a) gives production observability — all SSE/notify listener
    backends show up under ``aios-listener:<instance_id>`` in
    ``pg_stat_activity`` — and (b) lets the e2e leak test count ONLY this
    instance's listener backends, immune to other workers' / the pool's churn.
    """
    return await asyncpg.connect(
        normalize_dsn(db_url),
        server_settings={
            "application_name": listener_application_name(),
            **LISTENER_TCP_KEEPALIVE_SETTINGS,
        },
    )


@dataclass(slots=True)
class ListenSubscription:
    """A dedicated asyncpg connection LISTENing on one channel, with its
    delivery queue.

    Returned by the ``open_listen_for_*`` functions.  The caller (an SSE
    generator) owns calling :meth:`terminate` in a ``finally`` block to
    release the Postgres backend.  ``terminate`` is synchronous — safe to
    invoke under cancellation, where awaits would re-raise immediately
    (see the module docstring on ``conn.terminate()`` vs ``conn.close()``).
    """

    queue: asyncio.Queue[str]
    _conn: asyncpg.Connection[object]
    _release_capacity: Callable[[], None] | None = None

    def terminate(self) -> None:
        self._conn.terminate()
        if self._release_capacity is not None:
            self._release_capacity()


async def _open_drop_oldest_listener(
    db_url: str,
    channel: str,
    *,
    queue_max: int,
    log_key: str,
    log_fields: dict[str, str],
    on_connected: Callable[[asyncpg.Connection[object]], Awaitable[None]] | None = None,
) -> ListenSubscription:
    """Open a dedicated asyncpg connection LISTENing ``channel`` with a
    bounded **drop-oldest** delivery queue, and return a
    :class:`ListenSubscription`.

    This is the single source of the LISTEN-subscription open lifecycle that
    #502 ("close conn on add_listener failure") and #609 ("reap PG backends on
    disconnect via ``conn.terminate()``") previously had to shotgun-patch across
    five byte-identical copies. Every ``open_listen_for_*`` below is a thin
    call-through that supplies its channel template + log key/fields; a future
    backpressure/terminate-class fix lands here once.

    On queue overflow the callback drops the OLDEST queued payload to make room
    for the new one and logs ``log_key`` with ``log_fields`` (+ ``queue_max``).
    SSE consumers recover from drops by reconnecting with ``?after_seq=``.

    ``on_connected`` is an optional post-``add_listener`` hook run on the
    dedicated connection. It exists for exactly one caller —
    :func:`open_listen_for_events`, which passes ``acquire_subscriber_lock`` to
    take the #81 subscriber advisory lock — and is deliberately a documented,
    greppable hook rather than a bare ``lock: bool`` flag, so that a session's
    streaming-vs-non-streaming inference economics can't be silently flipped by
    toggling a boolean (see :func:`open_listen_for_events`).

    Any failure after the initial ``asyncpg.connect`` succeeds will
    ``conn.terminate()`` and re-raise.
    """
    release_capacity = _reserve_sse_subscriber()
    try:
        conn = await _connect_listener(db_url)
    except BaseException:
        if release_capacity is not None:
            release_capacity()
        raise
    try:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_max)

        def _callback(
            _conn: asyncpg.Connection[object],
            _pid: int,
            _channel: str,
            payload: str,
        ) -> None:
            # CRITICAL: this callback is invoked on asyncpg's connection read
            # loop. It MUST be synchronous. No await calls allowed here.
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning(log_key, queue_max=queue_max, **log_fields)
                # Drop oldest to make room. The SSE consumer can recover by
                # reconnecting with `?after_seq=`.
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

        await conn.add_listener(channel, _callback)
        if on_connected is not None:
            await on_connected(conn)
    except BaseException:
        conn.terminate()
        if release_capacity is not None:
            release_capacity()
        raise
    return ListenSubscription(queue=queue, _conn=conn, _release_capacity=release_capacity)


# Sentinel distinguishing "caller did not pass on_connected, use the default
# subscriber-lock hook" from "caller explicitly passed None to OMIT the lock".
# The await-poller passes ``on_connected=None`` to deliberately skip the #81
# lock (see below + services/sessions.py); a default of ``None`` would make
# that omission indistinguishable from the unspecified case.
_ACQUIRE_SUBSCRIBER_LOCK_DEFAULT = object()


async def open_listen_for_events(
    db_url: str,
    session_id: str,
    *,
    queue_max: int = 1000,
    on_connected: Callable[[asyncpg.Connection[object]], Awaitable[None]]
    | None
    | object = _ACQUIRE_SUBSCRIBER_LOCK_DEFAULT,
) -> ListenSubscription:
    """Open a dedicated asyncpg connection, LISTEN events_<session_id>,
    acquire the SSE subscriber lock, and return a :class:`ListenSubscription`.

    Two-phase counterpart to :func:`listen_for_events` for callers (SSE
    route handlers) that need to preflight setup BEFORE constructing the
    streaming response.

    By default this acquires the #81 subscriber advisory lock via the explicit
    ``on_connected=acquire_subscriber_lock`` hook: with the lock held, the
    worker's :func:`aios.db.sse_lock.has_subscriber` check returns True and the
    model call takes the streaming path (deltas → ``pg_notify`` → SSE clients).

    A caller that consumes ONLY the terminal completion state — never the
    deltas — should pass ``on_connected=None`` to OMIT the lock, so it doesn't
    force the awaited session's worker into the slower streaming path for a
    consumer that ignores deltas. The ``await``-primitive poller does exactly
    this, mirroring how :func:`open_listen_for_run_events` omits the lock
    entirely.

    The lock is a documented, greppable ``on_connected`` call-site — NOT a bare
    ``lock: bool`` flag — by design (issue #81): a future caller flipping a
    boolean would silently change a session's streaming-vs-non-streaming
    inference economics, whereas naming the hook keeps that coupling explicit
    and at exactly one site.

    Any failure after the initial ``asyncpg.connect`` succeeds will
    ``conn.terminate()`` and re-raise.
    """
    if on_connected is _ACQUIRE_SUBSCRIBER_LOCK_DEFAULT:
        # Hold a shared advisory lock on this dedicated connection so the
        # worker can detect that an SSE subscriber exists (issue #81).
        # pg_locks releases automatically on connection close — no cleanup.
        hook: Callable[[asyncpg.Connection[object]], Awaitable[None]] | None = functools.partial(
            acquire_subscriber_lock, session_id=session_id
        )
    else:
        hook = on_connected  # type: ignore[assignment]
    return await _open_drop_oldest_listener(
        db_url,
        f"events_{session_id}",
        queue_max=queue_max,
        log_key="listen.queue_full_drop",
        log_fields={"session_id": session_id},
        on_connected=hook,
    )


async def open_listen_for_run_events(
    db_url: str,
    run_id: str,
    *,
    queue_max: int = 1000,
) -> ListenSubscription:
    """Open a dedicated asyncpg connection LISTENing ``wf_run_events_<run_id>``.

    The workflow-run analog of :func:`open_listen_for_events` — same dedicated
    connection + bounded-queue-with-drop shape, feeding the ``/v1/runs/{id}/stream``
    generator. No subscriber advisory lock: that is a session-worker coordination
    signal (issue #81) with no workflow equivalent (a lost run wake self-heals via
    the needs-step sweep clauses, not via subscriber gating).
    """
    return await _open_drop_oldest_listener(
        db_url,
        f"wf_run_events_{run_id}",
        queue_max=queue_max,
        log_key="listen.wf_run_events_queue_full_drop",
        log_fields={"run_id": run_id},
    )


async def open_listen_for_connector_calls_by_type(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> ListenSubscription:
    """Open a dedicated asyncpg conn LISTENing ``connector_calls_<connector>``.

    Two-phase counterpart to :func:`listen_for_connector_calls_by_type`.
    """
    return await _open_drop_oldest_listener(
        db_url,
        f"connector_calls_{connector}",
        queue_max=queue_max,
        log_key="listen.connector_calls_type_queue_full_drop",
        log_fields={"connector": connector},
    )


async def open_listen_for_management_calls(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> ListenSubscription:
    """Open a dedicated asyncpg conn LISTENing ``connector_management_calls_<connector>``."""
    return await _open_drop_oldest_listener(
        db_url,
        f"connector_management_calls_{connector}",
        queue_max=queue_max,
        log_key="listen.connector_management_calls_queue_full_drop",
        log_fields={"connector": connector},
    )


async def open_listen_for_connection_discovery(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> ListenSubscription:
    """Open a dedicated asyncpg conn LISTENing ``connections_<connector>``."""
    return await _open_drop_oldest_listener(
        db_url,
        f"connections_{connector}",
        queue_max=queue_max,
        log_key="listen.connection_discovery_queue_full_drop",
        log_fields={"connector": connector},
    )


@asynccontextmanager
async def listen_for_connector_result(
    db_url: str,
    call_id: str,
) -> AsyncIterator[asyncio.Queue[str]]:
    """Open a dedicated asyncpg connection LISTENing on ``connector_result_<call_id>``.

    The API process uses this for the ``/v1/connectors/...`` endpoints:
    LISTEN first, enqueue the procrastinate task, await the single
    NOTIFY payload, return the parsed result.

    The yielded queue carries one entry per NOTIFY — for the connector
    RPC plane that's exactly one entry per call.  ``queue_max=8``
    leaves headroom for retries without blowing the queue.

    Mirrors :func:`listen_for_events` but with a per-call (not
    per-session) channel and a tighter queue bound.
    """
    conn = await _connect_listener(db_url)
    try:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
        channel = f"connector_result_{call_id}"

        def _callback(
            _conn: asyncpg.Connection[object],
            _pid: int,
            _channel: str,
            payload: str,
        ) -> None:
            # See ``listen_for_events`` for why this MUST be synchronous.
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning("listen.connector_result_queue_full", call_id=call_id)

        await conn.add_listener(channel, _callback)
        yield queue
    finally:
        conn.terminate()


@asynccontextmanager
async def _listen_subscription(
    open_coro: Awaitable[ListenSubscription],
) -> AsyncIterator[asyncio.Queue[str]]:
    """Generic context-manager wrapper over any ``open_listen_for_*`` coroutine.

    Single-sources the ``sub = await open_*()`` / ``yield sub.queue`` /
    ``finally: sub.terminate()`` skeleton shared by the named ``listen_for_*``
    wrappers below, so the terminate-on-exit guarantee lives in one place.
    """
    subscription = await open_coro
    try:
        yield subscription.queue
    finally:
        subscription.terminate()


def listen_for_events(
    db_url: str,
    session_id: str,
    *,
    queue_max: int = 1000,
) -> AbstractAsyncContextManager[asyncio.Queue[str]]:
    """Open a dedicated asyncpg connection, LISTEN events_<session_id>,
    yield an asyncio.Queue that receives event ids as they arrive.

    The queue is bounded. On overflow, the oldest payload is dropped to make
    room for the new one and a warning is logged. SSE clients can recover
    from drops by reconnecting with ``?after_seq=<their last seq>``.

    Thin wrapper over :func:`open_listen_for_events` for callers that
    prefer context-manager scoping.  SSE route handlers should use the
    ``open_*`` form directly so setup failures surface before
    ``EventSourceResponse`` writes 200 OK headers.

    Parameters
    ----------
    db_url:
        Postgres connection URL. Driver-prefix variants are normalized.
    session_id:
        The session whose events_<session_id> channel to LISTEN on.
    queue_max:
        Bound on the in-memory queue. Defaults to 1000 — generous enough
        that a slow consumer can fall behind by ~1000 events before drops.
    """
    return _listen_subscription(open_listen_for_events(db_url, session_id, queue_max=queue_max))


def listen_for_connector_calls_by_type(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> AbstractAsyncContextManager[asyncio.Queue[str]]:
    """LISTEN ``connector_calls_<connector>``; yield a queue of ``"<session_id>|<connection_id>"`` payloads.

    Used by the runtime SSE introduced in #328 PR 5: one runtime
    container subscribes once per ``connector`` type and fans out to
    its per-connection workers client-side via the ``connection_id``
    half of the payload.

    Thin wrapper over :func:`open_listen_for_connector_calls_by_type`.
    """
    return _listen_subscription(
        open_listen_for_connector_calls_by_type(db_url, connector, queue_max=queue_max)
    )


def listen_for_management_calls(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> AbstractAsyncContextManager[asyncio.Queue[str]]:
    """LISTEN ``connector_management_calls_<connector>``; yield a queue of ``call_id`` payloads.

    Thin wrapper over :func:`open_listen_for_management_calls`.
    """
    return _listen_subscription(
        open_listen_for_management_calls(db_url, connector, queue_max=queue_max)
    )


def listen_for_connection_discovery(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> AbstractAsyncContextManager[asyncio.Queue[str]]:
    """LISTEN ``connections_<connector>``; yield a queue of ``"<event>|<connection_id>|<account>"``.

    Backs the connection-discovery SSE (#328 PR 5). The emit side lives
    in :mod:`aios.services.connections.attach_connection` /
    ``archive_connection``.

    Thin wrapper over :func:`open_listen_for_connection_discovery`.
    """
    return _listen_subscription(
        open_listen_for_connection_discovery(db_url, connector, queue_max=queue_max)
    )


SESSION_INTERRUPT_CHANNEL = "aios_session_interrupt"


@asynccontextmanager
async def listen_for_session_interrupts(
    db_url: str,
) -> AsyncIterator[asyncio.Queue[str]]:
    """Yield session_id payloads from the interrupt channel.

    Interrupt NOTIFY delivery is fire-and-forget: an interrupt emitted while
    this dedicated LISTEN connection is disconnected or reconnecting is lost,
    because there is no durable interrupt row to backfill from. Silent drops
    are made detectable by TCP keepalive settings on the listener connection
    plus a termination listener that wakes the consumer; the worker-level
    reconnect loop then re-enters LISTEN after its backoff, and interrupts
    issued after reconnect dispatch normally.
    """
    conn = await _connect_listener(db_url)
    try:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1024)

        def _callback(
            _conn: asyncpg.Connection[object],
            _pid: int,
            _channel: str,
            payload: str,
        ) -> None:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning("listen.session_interrupt_queue_full")

        def _termination_callback(_conn: asyncpg.Connection[object]) -> None:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait("")

        conn.add_termination_listener(_termination_callback)
        await conn.add_listener(SESSION_INTERRUPT_CHANNEL, _callback)
        yield queue
    finally:
        conn.terminate()


MCP_EVICT_VAULT_CHANNEL = "aios_mcp_evict_vault"


@asynccontextmanager
async def listen_for_mcp_evict_vault(
    db_url: str,
) -> AsyncIterator[asyncio.Queue[str]]:
    """Yield ``vault_id`` payloads from the MCP-pool eviction channel.

    An operator credential mutation (PUT/archive/delete on a vault
    credential, plus vault archive/delete) runs in the API process and
    fires a NOTIFY on this channel; the worker — which owns the MCP session
    pool — LISTENs here and evicts the pooled sessions keyed on that vault
    so the rotated secret propagates immediately instead of waiting out the
    900s idle TTL that active use defeats (#1030).

    Delivery is fire-and-forget (no durable backfill row), exactly like
    :func:`listen_for_session_interrupts`: an eviction emitted while this
    dedicated LISTEN connection is reconnecting is lost, but the idle reaper
    remains the safety net and a subsequent mutation re-fires. Silent drops
    are made detectable by TCP keepalive plus a termination listener that
    wakes the consumer into the worker-level reconnect loop.
    """
    conn = await _connect_listener(db_url)
    try:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1024)

        def _callback(
            _conn: asyncpg.Connection[object],
            _pid: int,
            _channel: str,
            payload: str,
        ) -> None:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning("listen.mcp_evict_vault_queue_full")

        def _termination_callback(_conn: asyncpg.Connection[object]) -> None:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait("")

        conn.add_termination_listener(_termination_callback)
        await conn.add_listener(MCP_EVICT_VAULT_CHANNEL, _callback)
        yield queue
    finally:
        conn.terminate()


GITHUB_CLONE_BREAKER_CLEAR_CHANNEL = "aios_github_clone_breaker_clear"


@asynccontextmanager
async def listen_for_github_clone_breaker_clear(
    db_url: str,
) -> AsyncIterator[asyncio.Queue[str]]:
    """Yield ``resource_id`` payloads from the github-clone-breaker clear channel.

    A token rotation (``PUT /v1/sessions/{id}/resources/{resource_id}``, which
    calls :func:`aios.services.github_repositories.rotate_token`) runs in the
    API process and fires a NOTIFY on this channel with the ``ghrepo_...``
    resource id; the worker — which owns the in-memory
    :class:`aios.sandbox.github_clone_breaker.GithubCloneBreaker` — LISTENs
    here and clears that resource's breaker state so a fixed credential
    re-probes on the very next provision instead of serving out a cooldown
    opened under the old secret (#1720, re-probe path (a)).

    Delivery is fire-and-forget, exactly like
    :func:`listen_for_mcp_evict_vault`: a clear-signal emitted while this
    dedicated LISTEN connection is reconnecting is lost, but the breaker's
    own cooldown-then-half-open-probe re-probe (path (c)) is the safety net,
    and a subsequent rotation re-fires.
    """
    conn = await _connect_listener(db_url)
    try:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1024)

        def _callback(
            _conn: asyncpg.Connection[object],
            _pid: int,
            _channel: str,
            payload: str,
        ) -> None:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning("listen.github_clone_breaker_clear_queue_full")

        def _termination_callback(_conn: asyncpg.Connection[object]) -> None:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait("")

        conn.add_termination_listener(_termination_callback)
        await conn.add_listener(GITHUB_CLONE_BREAKER_CLEAR_CHANNEL, _callback)
        yield queue
    finally:
        conn.terminate()


# Channel VALUE is byte-identical across the #818 rename (the underlying
# Postgres NOTIFY trigger function ``notify_scheduled_tasks_due`` and its
# channel string stay put — renaming buys nothing and opens a deploy window
# where an old worker listens on a channel nothing notifies). Only the
# Python helper below is renamed to the new ``triggers`` vocabulary.
SCHEDULED_TASKS_DUE_CHANNEL = "aios_scheduled_tasks_due"


@asynccontextmanager
async def listen_for_triggers_due(
    db_url: str,
) -> AsyncIterator[asyncio.Event]:
    """Yield an :class:`asyncio.Event` that fires whenever the ``triggers``
    NOTIFY trigger emits.

    The scheduler doesn't care which row changed — it always recomputes
    ``MIN(next_fire)`` on wake — so this listener collapses all incoming
    notifications onto a single :class:`asyncio.Event`. The scheduler's
    loop ``await``s ``asyncio.wait_for(event.wait(), timeout=...)``; the
    timeout is the cold-path heartbeat for connection resilience.

    The event is cleared inside this context manager before yielding, so
    callers see a clean edge for each wake.
    """
    conn = await _connect_listener(db_url)
    try:
        event = asyncio.Event()

        def _callback(
            _conn: asyncpg.Connection[object],
            _pid: int,
            _channel: str,
            _payload: str,
        ) -> None:
            # See ``listen_for_events`` for why this MUST be synchronous.
            event.set()

        await conn.add_listener(SCHEDULED_TASKS_DUE_CHANNEL, _callback)
        yield event
    finally:
        conn.terminate()
