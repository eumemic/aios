"""SSE event stream generator.

Used by ``GET /v1/sessions/{id}/stream``. The generator:

1. Opens a dedicated asyncpg LISTEN connection (via
   :func:`aios.db.listen.listen_for_events`) BEFORE issuing the backfill
   SELECT — see the LISTEN-before-backfill discussion in ``db/listen.py``.
2. Backfills events from ``after_seq``, tracking the highest seq seen as
   ``cursor``.
3. Tails live notifications. For each notify, fetches the matching event by
   id and emits it ONLY if its seq exceeds ``cursor`` (dedup against the
   backfill).
4. Detects ``terminated`` lifecycle events and emits a ``done`` SSE event
   before exiting.

Client disconnect handling is automatic: sse-starlette's
``EventSourceResponse`` raises ``asyncio.CancelledError`` into the generator
when the client closes the connection. The ``finally`` blocks in
``listen_for_events`` and the pool acquires clean up. Don't trap
CancelledError; let it propagate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import asyncpg
from sse_starlette import ServerSentEvent

from aios.db import queries
from aios.db.listen import listen_for_events
from aios.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger("aios.api.sse")


def _event_to_sse(event_dict: dict[str, Any]) -> ServerSentEvent:
    """Build a ServerSentEvent from an event dict.

    The data field is JSON. The SSE event name is just "event" — clients
    can branch on the inner ``kind`` field.
    """
    import json

    return ServerSentEvent(data=json.dumps(event_dict), event="event")


def _serialize_event(row: asyncpg.Record) -> dict[str, Any]:
    """Convert an asyncpg event row to a JSON-friendly dict.

    ``orig_channel`` and ``channel`` come along so downstream consumers
    (e.g. the ``aios tail`` CLI) can tag user messages and spot
    wrong-channel sends without re-querying.
    """
    import json

    raw_data = row["data"]
    parsed = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "seq": row["seq"],
        "kind": row["kind"],
        "data": parsed,
        "created_at": row["created_at"].isoformat(),
        "orig_channel": row["orig_channel"],
        "channel": row["channel"],
    }


async def sse_event_stream(
    db_url: str,
    pool: asyncpg.Pool[Any],
    session_id: str,
    after_seq: int = 0,
) -> AsyncIterator[ServerSentEvent]:
    """Yield ServerSentEvents for a session, backfilling from ``after_seq``
    and then tailing live notifications.

    Critical ordering: open the LISTEN connection BEFORE running the backfill
    SELECT. Otherwise events that commit during the backfill window are lost
    (the SELECT can't see uncommitted-at-snapshot rows, and the listener
    isn't yet attached to receive their NOTIFY).
    """
    async with listen_for_events(db_url, session_id) as queue:
        cursor = after_seq

        # Backfill: read all events with seq > after_seq, in order.
        async with pool.acquire() as conn:
            backfill_rows = await conn.fetch(
                "SELECT * FROM events WHERE session_id = $1 AND seq > $2 ORDER BY seq ASC",
                session_id,
                after_seq,
            )

        for row in backfill_rows:
            payload = _serialize_event(row)
            cursor = max(cursor, payload["seq"])
            yield _event_to_sse(payload)
            if _is_terminal(payload):
                yield ServerSentEvent(data="{}", event="done")
                return

        # Tail live notifications. The dedup against `cursor` covers the
        # rare case where an event is in BOTH the backfill (because it
        # committed before the SELECT's snapshot) AND the queue (because
        # NOTIFY fired during the SELECT).
        while True:
            notification = await queue.get()

            # Transient streaming deltas (no DB row). The payload is
            # JSON like {"delta":"He"} — distinguishable from event IDs
            # (which start with "evt_") by the leading "{".
            if notification.startswith("{"):
                yield ServerSentEvent(data=notification, event="delta")
                continue

            event_id = notification
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM events WHERE id = $1", event_id)
            if row is None:
                # Shouldn't happen — events are immutable. Log and skip.
                log.warning("sse.event_not_found", event_id=event_id)
                continue
            event_data = _serialize_event(row)
            if event_data["seq"] <= cursor:
                continue
            cursor = event_data["seq"]
            yield _event_to_sse(event_data)
            if _is_terminal(event_data):
                yield ServerSentEvent(data="{}", event="done")
                return


def _is_terminal(payload: dict[str, Any]) -> bool:
    """True if this event marks the session as fully terminated.

    A `lifecycle` event with `status: terminated` ends the SSE stream. We
    don't end on `idle` because the client may want to keep watching for
    follow-up turns triggered by future user messages.
    """
    return (
        payload["kind"] == "lifecycle"
        and isinstance(payload["data"], dict)
        and payload["data"].get("status") == "terminated"
    )


# Reference imports for static analysis: queries is reserved for future use
# (e.g., per-session validation before opening the stream).
_ = queries
