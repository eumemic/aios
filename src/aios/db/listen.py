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
a fresh connection per SSE client. The connection is closed cleanly in the
context manager's ``finally`` block, which implicitly issues ``UNLISTEN``.

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
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import asyncpg

from aios.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger("aios.db.listen")


def _normalize_dsn(db_url: str) -> str:
    """Strip driver-suffix prefixes; asyncpg.connect wants bare postgresql://."""
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if db_url.startswith(prefix):
            return "postgresql://" + db_url[len(prefix) :]
    return db_url


@asynccontextmanager
async def listen_for_events(
    db_url: str,
    session_id: str,
    *,
    queue_max: int = 1000,
) -> AsyncIterator[asyncio.Queue[str]]:
    """Open a dedicated asyncpg connection, LISTEN events_<session_id>,
    yield an asyncio.Queue that receives event ids as they arrive.

    On exit, the connection is closed (UNLISTEN is implicit at close).

    The queue is bounded. On overflow, the oldest payload is dropped to make
    room for the new one and a warning is logged. SSE clients can recover
    from drops by reconnecting with ``?after_seq=<their last seq>``.

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
    conn = await asyncpg.connect(_normalize_dsn(db_url))
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
    try:
        yield queue
    finally:
        with contextlib.suppress(Exception):
            await conn.remove_listener(channel, _callback)
        with contextlib.suppress(Exception):
            await conn.close()
