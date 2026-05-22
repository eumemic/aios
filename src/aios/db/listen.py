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
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import asyncpg

from aios.db.pool import normalize_dsn
from aios.db.sse_lock import acquire_subscriber_lock
from aios.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger("aios.db.listen")


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

    def terminate(self) -> None:
        self._conn.terminate()


async def open_listen_for_events(
    db_url: str,
    session_id: str,
    *,
    queue_max: int = 1000,
) -> ListenSubscription:
    """Open a dedicated asyncpg connection, LISTEN events_<session_id>,
    acquire the SSE subscriber lock, and return a :class:`ListenSubscription`.

    Two-phase counterpart to :func:`listen_for_events` for callers (SSE
    route handlers) that need to preflight setup BEFORE constructing the
    streaming response.

    Any failure after the initial ``asyncpg.connect`` succeeds will
    ``conn.terminate()`` and re-raise.
    """
    conn = await asyncpg.connect(normalize_dsn(db_url))
    try:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_max)
        channel = f"events_{session_id}"

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
                log.warning(
                    "listen.queue_full_drop",
                    session_id=session_id,
                    queue_max=queue_max,
                )
                # Drop oldest to make room. The SSE consumer can recover by
                # reconnecting with `?after_seq=`.
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

        await conn.add_listener(channel, _callback)
        # Hold a shared advisory lock on this dedicated connection so the
        # worker can detect that an SSE subscriber exists (issue #81).
        # pg_locks releases automatically on connection close — no cleanup.
        await acquire_subscriber_lock(conn, session_id)
    except BaseException:
        conn.terminate()
        raise
    return ListenSubscription(queue=queue, _conn=conn)


async def open_listen_for_connector_calls_by_type(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> ListenSubscription:
    """Open a dedicated asyncpg conn LISTENing ``connector_calls_<connector>``.

    Two-phase counterpart to :func:`listen_for_connector_calls_by_type`.
    """
    conn = await asyncpg.connect(normalize_dsn(db_url))
    try:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_max)
        channel = f"connector_calls_{connector}"

        def _callback(
            _conn: asyncpg.Connection[object],
            _pid: int,
            _channel: str,
            payload: str,
        ) -> None:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning(
                    "listen.connector_calls_type_queue_full_drop",
                    connector=connector,
                    queue_max=queue_max,
                )
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

        await conn.add_listener(channel, _callback)
    except BaseException:
        conn.terminate()
        raise
    return ListenSubscription(queue=queue, _conn=conn)


async def open_listen_for_management_calls(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> ListenSubscription:
    """Open a dedicated asyncpg conn LISTENing ``connector_management_calls_<connector>``."""
    conn = await asyncpg.connect(normalize_dsn(db_url))
    try:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_max)
        channel = f"connector_management_calls_{connector}"

        def _callback(
            _conn: asyncpg.Connection[object],
            _pid: int,
            _channel: str,
            payload: str,
        ) -> None:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning(
                    "listen.connector_management_calls_queue_full_drop",
                    connector=connector,
                    queue_max=queue_max,
                )
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

        await conn.add_listener(channel, _callback)
    except BaseException:
        conn.terminate()
        raise
    return ListenSubscription(queue=queue, _conn=conn)


async def open_listen_for_connection_discovery(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> ListenSubscription:
    """Open a dedicated asyncpg conn LISTENing ``connections_<connector>``."""
    conn = await asyncpg.connect(normalize_dsn(db_url))
    try:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_max)
        channel = f"connections_{connector}"

        def _callback(
            _conn: asyncpg.Connection[object],
            _pid: int,
            _channel: str,
            payload: str,
        ) -> None:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning(
                    "listen.connection_discovery_queue_full_drop",
                    connector=connector,
                    queue_max=queue_max,
                )
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

        await conn.add_listener(channel, _callback)
    except BaseException:
        conn.terminate()
        raise
    return ListenSubscription(queue=queue, _conn=conn)


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
    conn = await asyncpg.connect(normalize_dsn(db_url))
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
async def listen_for_events(
    db_url: str,
    session_id: str,
    *,
    queue_max: int = 1000,
) -> AsyncIterator[asyncio.Queue[str]]:
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
    subscription = await open_listen_for_events(db_url, session_id, queue_max=queue_max)
    try:
        yield subscription.queue
    finally:
        subscription.terminate()


@asynccontextmanager
async def listen_for_connector_calls_by_type(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> AsyncIterator[asyncio.Queue[str]]:
    """LISTEN ``connector_calls_<connector>``; yield a queue of ``"<session_id>|<connection_id>"`` payloads.

    Used by the runtime SSE introduced in #328 PR 5: one runtime
    container subscribes once per ``connector`` type and fans out to
    its per-connection workers client-side via the ``connection_id``
    half of the payload.

    Thin wrapper over :func:`open_listen_for_connector_calls_by_type`.
    """
    subscription = await open_listen_for_connector_calls_by_type(
        db_url, connector, queue_max=queue_max
    )
    try:
        yield subscription.queue
    finally:
        subscription.terminate()


@asynccontextmanager
async def listen_for_management_calls(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> AsyncIterator[asyncio.Queue[str]]:
    """LISTEN ``connector_management_calls_<connector>``; yield a queue of ``call_id`` payloads.

    Thin wrapper over :func:`open_listen_for_management_calls`.
    """
    subscription = await open_listen_for_management_calls(db_url, connector, queue_max=queue_max)
    try:
        yield subscription.queue
    finally:
        subscription.terminate()


@asynccontextmanager
async def listen_for_connection_discovery(
    db_url: str,
    connector: str,
    *,
    queue_max: int = 1000,
) -> AsyncIterator[asyncio.Queue[str]]:
    """LISTEN ``connections_<connector>``; yield a queue of ``"<event>|<connection_id>|<account>"``.

    Backs the connection-discovery SSE (#328 PR 5). The emit side lives
    in :mod:`aios.services.connections.attach_connection` /
    ``archive_connection``.

    Thin wrapper over :func:`open_listen_for_connection_discovery`.
    """
    subscription = await open_listen_for_connection_discovery(
        db_url, connector, queue_max=queue_max
    )
    try:
        yield subscription.queue
    finally:
        subscription.terminate()


SESSION_INTERRUPT_CHANNEL = "aios_session_interrupt"


@asynccontextmanager
async def listen_for_session_interrupts(
    db_url: str,
) -> AsyncIterator[asyncio.Queue[str]]:
    """Yield a queue of session_id payloads from the interrupt channel."""
    conn = await asyncpg.connect(normalize_dsn(db_url))
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

        await conn.add_listener(SESSION_INTERRUPT_CHANNEL, _callback)
        yield queue
    finally:
        conn.terminate()
