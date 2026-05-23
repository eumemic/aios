"""SSE event stream generators.

Used by ``GET /v1/sessions/{id}/stream`` and the runtime-facing SSE
endpoints in ``aios.api.routers.connectors``.  Each generator:

1. Accepts a pre-established :class:`aios.db.listen.ListenSubscription`
   — the route handler opens it via ``open_listen_for_*`` BEFORE
   constructing :class:`sse_starlette.EventSourceResponse`, so a setup
   failure surfaces as a clean 503 instead of a half-open chunked stream
   (issue #376).
2. Backfills from the DB, tracking a cursor where ordering matters
   (sessions stream).
3. Tails live notifications from ``subscription.queue``.
4. Owns terminating the subscription in ``finally`` — under cancellation
   (client disconnect) AND under any in-body exception.

Critical ordering invariant: the route handler must run ``LISTEN`` (i.e.
call ``open_listen_for_*``) BEFORE the generator issues its backfill
``SELECT``.  See the discussion in ``aios.db.listen``.

Client disconnect handling: sse-starlette's ``EventSourceResponse`` raises
``asyncio.CancelledError`` into the generator when the client closes the
connection.  The ``finally`` block's ``subscription.terminate()`` is
synchronous and safe under cancellation.  Don't trap ``CancelledError``;
let it propagate.
"""

from __future__ import annotations

import json
import weakref
from typing import TYPE_CHECKING, Any

import asyncpg
from sse_starlette import EventSourceResponse, ServerSentEvent

from aios.db import queries
from aios.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aios.db.listen import ListenSubscription

log = get_logger("aios.api.sse")

# Realistic transient failures during testcontainer Postgres warmup or
# socket churn — these surface as a clean 503 from the SSE route handlers
# (issue #376). Anything else bubbles as an unhandled 500.
SSE_PREFLIGHT_EXCEPTIONS = (asyncpg.PostgresError, OSError)


def make_sse_response(
    subscription: ListenSubscription,
    content: AsyncIterator[ServerSentEvent],
    *,
    ping: int = 15,
) -> EventSourceResponse:
    """Build an ``EventSourceResponse`` whose ``ListenSubscription`` is
    cleaned up even on paths that never iterate the wrapped generator.

    Normal lifecycle: ``__call__`` runs, the generator iterates, its
    ``finally: subscription.terminate()`` fires on consumer disconnect
    or stream end.

    Pathological lifecycle (the bug this exists to close): the request
    task is cancelled between FastAPI's ``response = await handler()``
    and the subsequent ``await response(scope, receive, send)`` (e.g.
    server shutdown striking mid-dispatch, or a fast client disconnect).
    ``__call__`` never runs, the async generator is never started — and
    Python's async-generator semantics guarantee that an
    *unstarted* generator's ``finally`` block does NOT execute on
    ``aclose`` or GC. Without a backup, the dedicated asyncpg
    connection, its ``LISTEN``, and the SSE subscriber advisory lock
    leak until TCP keepalive reaps the backend (~2h).

    A ``weakref.finalize`` registers ``subscription.terminate`` to fire
    when the response is GC'd, catching the un-invoked path. In the
    normal path it fires *after* the generator's own ``finally`` has
    already terminated the connection — ``terminate`` is idempotent at
    asyncpg's transport layer (``transport.abort`` and the protocol's
    ``_state`` transition both tolerate repeated calls), so the second
    call is a no-op.
    """
    response = EventSourceResponse(content, ping=ping)
    weakref.finalize(response, subscription.terminate)
    return response


def _event_to_sse(event_dict: dict[str, Any]) -> ServerSentEvent:
    """Build a ServerSentEvent from an event dict.

    The data field is JSON. The SSE event name is just "event" — clients
    can branch on the inner ``kind`` field.
    """
    return ServerSentEvent(data=json.dumps(event_dict), event="event")


def _serialize_event(row: asyncpg.Record) -> dict[str, Any]:
    """Convert an asyncpg event row to a JSON-friendly dict.

    ``orig_channel`` and ``channel`` come along so downstream consumers
    (e.g. the ``aios tail`` CLI) can tag user messages and spot
    wrong-channel sends without re-querying.
    """
    parsed = queries.parse_jsonb(row["data"])
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
    subscription: ListenSubscription,
    pool: asyncpg.Pool[Any],
    session_id: str,
    after_seq: int = 0,
) -> AsyncIterator[ServerSentEvent]:
    """Yield ServerSentEvents for a session, backfilling from ``after_seq``
    and then tailing live notifications.

    The ``subscription`` must have been opened by the route handler via
    ``open_listen_for_events`` before this generator runs — the
    LISTEN-before-backfill invariant requires it.  This generator owns
    terminating the subscription in ``finally``.
    """
    queue = subscription.queue
    try:
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
    except Exception:
        log.exception("sse.session_events.body_raised", session_id=session_id)
        raise
    finally:
        subscription.terminate()


async def runtime_connector_calls_stream(
    subscription: ListenSubscription,
    pool: asyncpg.Pool[Any],
    connector: str,
    *,
    account_id: str,
    connection_ids: list[str] | None = None,
) -> AsyncIterator[ServerSentEvent]:
    """Yield SSE events for pending custom tool calls across every active
    connection of ``connector`` type (#328 PR 5).

    The bearer scopes the caller to one connector *type*; the emitted
    payload includes a ``connection_id`` field so the runtime container
    can fan out to its per-connection workers client-side.

    Backfills any pending calls at subscribe time, then tails the
    ``connector_calls_<connector>`` NOTIFY channel.

    When ``connection_ids`` is non-``None`` (#350), backfill and tail
    both filter to that allowlist — out-of-scope calls are silently
    omitted.  The tail-side check parses the NOTIFY payload directly
    so the per-connection fetch is skipped for out-of-scope IDs.
    """
    allowlist = set(connection_ids) if connection_ids is not None else None
    queue = subscription.queue
    try:
        emitted: set[str] = set()

        async with pool.acquire() as conn:
            backfill = await queries.list_pending_calls_for_connector(
                conn, connector, account_id=account_id
            )
        for call in backfill:
            if allowlist is not None and call.get("connection_id") not in allowlist:
                continue
            emitted.add(call["tool_call_id"])
            yield ServerSentEvent(data=json.dumps(call), event="call")

        while True:
            payload = await queue.get()
            # Payload format: "<session_id>|<connection_id>".
            session_id, _, connection_id = payload.partition("|")
            if not connection_id:
                log.warning("sse.runtime_calls.malformed_payload", payload=payload)
                continue
            # Skip the per-connection fetch for out-of-scope IDs.
            if allowlist is not None and connection_id not in allowlist:
                continue
            async with pool.acquire() as conn:
                pending = await queries.list_pending_calls_for_session_and_connection(
                    conn,
                    session_id=session_id,
                    connection_id=connection_id,
                    account_id=account_id,
                )
            for call in pending:
                if call["tool_call_id"] in emitted:
                    continue
                emitted.add(call["tool_call_id"])
                call["connection_id"] = connection_id
                yield ServerSentEvent(data=json.dumps(call), event="call")
    except Exception:
        log.exception("sse.runtime_calls.body_raised", connector=connector)
        raise
    finally:
        subscription.terminate()


async def management_calls_stream(
    subscription: ListenSubscription,
    pool: asyncpg.Pool[Any],
    connector: str,
    *,
    account_id: str,
) -> AsyncIterator[ServerSentEvent]:
    """Yield SSE events for pending management calls of ``connector`` type.

    Backfills pending unexpired calls, then tails
    ``connector_management_calls_<connector>``.  Each event:
    ``{"call_id": "mgmt_...", "method": str, "params": dict}``.
    """
    queue = subscription.queue
    try:
        emitted: set[str] = set()

        async with pool.acquire() as conn:
            backfill = await queries.list_pending_management_calls_for_connector(
                conn, connector, account_id=account_id
            )
        for call in backfill:
            emitted.add(call["call_id"])
            yield ServerSentEvent(data=json.dumps(call), event="call")

        while True:
            call_id = await queue.get()
            if call_id in emitted:
                continue
            # The NOTIFY payload is just the id; re-fetch the row for the
            # method + params.  Keeping the NOTIFY payload tiny stays well
            # under the 8000-byte cap and means a follow-up UPDATE can't
            # desync from a stale payload already in flight.
            async with pool.acquire() as conn:
                row = await queries.get_management_call(conn, call_id, account_id=account_id)
            if row is None or row["status"] != "pending":
                continue
            emitted.add(call_id)
            yield ServerSentEvent(
                data=json.dumps(
                    {"call_id": row["id"], "method": row["method"], "params": row["params"]}
                ),
                event="call",
            )
    except Exception:
        log.exception("sse.management_calls.body_raised", connector=connector)
        raise
    finally:
        subscription.terminate()


async def connection_discovery_stream(
    subscription: ListenSubscription,
    pool: asyncpg.Pool[Any],
    connector: str,
    *,
    account_id: str,
    connection_ids: list[str] | None = None,
) -> AsyncIterator[ServerSentEvent]:
    """Yield ``added`` SSE events backfilling every active connection of
    ``connector`` type, then ``added`` / ``removed`` events as connections
    are attached or archived (#328 PR 5).

    Backs the runtime container's connection-discovery loop: it
    subscribes once, walks the ``added`` events to spawn per-connection
    workers, and tears workers down on ``removed`` events.  The emit
    side lives in :mod:`aios.services.connections.attach_connection` /
    ``archive_connection``.

    When ``connection_ids`` is non-``None`` (#350), the backfill and
    tail both filter to that allowlist — out-of-scope IDs are silently
    skipped in either phase so the runtime container's discovery loop
    just doesn't see them.
    """
    allowlist = set(connection_ids) if connection_ids is not None else None
    queue = subscription.queue
    try:
        emitted_added: set[str] = set()

        # Page through all active connections of this type; the default
        # ``list_connections`` limit (50) would silently under-fanout for
        # runtimes with more than 50 active connections.
        cursor: str | None = None
        while True:
            async with pool.acquire() as conn:
                page = await queries.list_connections(
                    conn, connector=connector, limit=200, after=cursor, account_id=account_id
                )
            for connection in page:
                if allowlist is not None and connection.id not in allowlist:
                    continue
                emitted_added.add(connection.id)
                yield ServerSentEvent(
                    data=json.dumps(
                        {
                            "event": "added",
                            "connection_id": connection.id,
                            "external_account_id": connection.external_account_id,
                        }
                    ),
                    event="connection",
                )
            if len(page) < 200:
                break
            cursor = page[-1].id

        while True:
            payload = await queue.get()
            # Payload format: "<event>|<connection_id>|<event_account_id>|<external_account_id>".
            parts = payload.split("|", 3)
            if len(parts) != 4:
                log.warning("sse.discovery.malformed_payload", payload=payload)
                continue
            event, connection_id, event_account_id, external_account_id = parts
            # Tenant isolation: a runtime token scopes a subscriber to one
            # tenant; dropping cross-tenant NOTIFY events here closes an
            # existence-leak (sibling account_ids would otherwise surface
            # via the tail).
            if event_account_id != account_id:
                continue
            if allowlist is not None and connection_id not in allowlist:
                continue
            if event == "added" and connection_id in emitted_added:
                continue
            if event == "added":
                emitted_added.add(connection_id)
            elif event == "removed":
                emitted_added.discard(connection_id)
            yield ServerSentEvent(
                data=json.dumps(
                    {
                        "event": event,
                        "connection_id": connection_id,
                        "external_account_id": external_account_id,
                    }
                ),
                event="connection",
            )
    except Exception:
        log.exception("sse.connection_discovery.body_raised", connector=connector)
        raise
    finally:
        subscription.terminate()


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
