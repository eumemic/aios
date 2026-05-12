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
from aios.db.listen import (
    listen_for_connection_discovery,
    listen_for_connector_calls,
    listen_for_connector_calls_by_type,
    listen_for_events,
)
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


async def connector_calls_stream(
    db_url: str,
    pool: asyncpg.Pool[Any],
    connection_id: str,
) -> AsyncIterator[ServerSentEvent]:
    """Yield SSE events for pending custom tool calls owned by ``connection_id``.

    Backfills any pending calls at subscribe time, then tails the
    ``connector_calls_<connection_id>`` NOTIFY channel.  When NOTIFY
    fires (with a ``session_id`` payload), looks up that session's
    pending tool calls and emits one SSE event per call.

    Each emitted event is keyed ``"call"`` with a JSON body shaped::

        {
            "session_id": "sess_xxx",
            "tool_call_id": "call_yyy",
            "name": "telegram_send",
            "arguments": "{...}",
            "focal_channel": "telegram/bot1/chat123" | null
        }

    The connector dedupes by ``tool_call_id`` client-side so SSE
    reconnects (which replay the backfill) don't double-execute.
    """
    import json as _json

    async with listen_for_connector_calls(db_url, connection_id) as queue:
        emitted: set[str] = set()

        async with pool.acquire() as conn:
            backfill = await queries.list_pending_calls_for_connection(conn, connection_id)
        for call in backfill:
            emitted.add(call["tool_call_id"])
            yield ServerSentEvent(data=_json.dumps(call), event="call")

        while True:
            session_id = await queue.get()
            async with pool.acquire() as conn:
                pending = await queries.list_pending_calls_for_session_and_connection(
                    conn,
                    session_id=session_id,
                    connection_id=connection_id,
                )
            for call in pending:
                if call["tool_call_id"] in emitted:
                    continue
                emitted.add(call["tool_call_id"])
                yield ServerSentEvent(data=_json.dumps(call), event="call")


async def runtime_connector_calls_stream(
    db_url: str,
    pool: asyncpg.Pool[Any],
    connector: str,
) -> AsyncIterator[ServerSentEvent]:
    """Yield SSE events for pending custom tool calls across every active
    connection of ``connector`` type (#328 PR 5).

    Counterpart to :func:`connector_calls_stream`: that one is scoped to
    a single ``connection_id`` (legacy per-connection auth); this one is
    scoped to a connector *type* and the emitted payload includes a
    ``connection_id`` field so the runtime container can fan out to its
    per-connection workers client-side.

    Backfills any pending calls at subscribe time, then tails the
    ``connector_calls_<connector>`` NOTIFY channel.
    """
    import json as _json

    async with listen_for_connector_calls_by_type(db_url, connector) as queue:
        emitted: set[str] = set()

        async with pool.acquire() as conn:
            backfill = await queries.list_pending_calls_for_connector(conn, connector)
        for call in backfill:
            emitted.add(call["tool_call_id"])
            yield ServerSentEvent(data=_json.dumps(call), event="call")

        while True:
            payload = await queue.get()
            # Payload format: "<session_id>|<connection_id>".
            session_id, _, connection_id = payload.partition("|")
            if not connection_id:
                log.warning("sse.runtime_calls.malformed_payload", payload=payload)
                continue
            async with pool.acquire() as conn:
                pending = await queries.list_pending_calls_for_session_and_connection(
                    conn,
                    session_id=session_id,
                    connection_id=connection_id,
                )
            for call in pending:
                if call["tool_call_id"] in emitted:
                    continue
                emitted.add(call["tool_call_id"])
                call["connection_id"] = connection_id
                yield ServerSentEvent(data=_json.dumps(call), event="call")


async def connection_discovery_stream(
    db_url: str,
    pool: asyncpg.Pool[Any],
    connector: str,
) -> AsyncIterator[ServerSentEvent]:
    """Yield ``added`` SSE events backfilling every active connection of
    ``connector`` type, then ``added`` / ``removed`` events as connections
    are attached or archived (#328 PR 5).

    Backs the runtime container's connection-discovery loop: it
    subscribes once, walks the ``added`` events to spawn per-connection
    workers, and tears workers down on ``removed`` events.  The emit
    side lives in :mod:`aios.services.connections.attach_connection` /
    ``archive_connection``.
    """
    import json as _json

    async with listen_for_connection_discovery(db_url, connector) as queue:
        emitted_added: set[str] = set()

        async with pool.acquire() as conn:
            backfill = await queries.list_connections(conn, connector=connector)
        for connection in backfill:
            emitted_added.add(connection.id)
            yield ServerSentEvent(
                data=_json.dumps(
                    {
                        "event": "added",
                        "connection_id": connection.id,
                        "account": connection.account,
                    }
                ),
                event="connection",
            )

        while True:
            payload = await queue.get()
            # Payload format: "<event>|<connection_id>|<account>".
            parts = payload.split("|", 2)
            if len(parts) != 3:
                log.warning("sse.discovery.malformed_payload", payload=payload)
                continue
            event, connection_id, account = parts
            if event == "added" and connection_id in emitted_added:
                continue
            if event == "added":
                emitted_added.add(connection_id)
            elif event == "removed":
                emitted_added.discard(connection_id)
            yield ServerSentEvent(
                data=_json.dumps(
                    {
                        "event": event,
                        "connection_id": connection_id,
                        "account": account,
                    }
                ),
                event="connection",
            )


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
