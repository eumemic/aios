"""Cancel-supervision side-table queries (cancel-design §2).

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared contract. The durable primitive the recursive ``cancel_invocation`` cascade is
built on: the session-side ``session_cancel_markers`` exit-marker (the run side reuses
``wf_run_signals kind='cancel'`` verbatim). Every write is to a **side-table**, never to
a node's event log/journal — the single-writer invariant: a node's own step under its
own lock is the only writer of its log.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.models.invocations import SessionCancelMarker


def _row_to_session_cancel_marker(row: asyncpg.Record) -> SessionCancelMarker:
    return SessionCancelMarker(
        session_id=row["session_id"],
        request_id=row["request_id"],
        account_id=row["account_id"],
        harvested_at=row["harvested_at"],
        created_at=row["created_at"],
    )


# ─── session_cancel_markers exit-marker ──────────────────────────────────────


async def insert_session_cancel_marker(
    conn: asyncpg.Connection[Any], *, session_id: str, request_id: str, account_id: str
) -> bool:
    """Seed a cancel-marker on a session edge; ``True`` iff newly inserted.

    ``ON CONFLICT DO NOTHING`` is the cascade's idempotency (re-propagation / a repeated
    cancel is a no-op): ``RETURNING`` yields the row only on a fresh insert, so the bool
    reports a genuinely-new mark vs a no-op re-seed.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO session_cancel_markers (session_id, request_id, account_id)
        VALUES ($1, $2, $3)
        ON CONFLICT (session_id, request_id) DO NOTHING
        RETURNING session_id
        """,
        session_id,
        request_id,
        account_id,
    )
    return row is not None


async def get_session_cancel_marker(
    conn: asyncpg.Connection[Any], *, session_id: str, request_id: str
) -> SessionCancelMarker | None:
    row = await conn.fetchrow(
        "SELECT * FROM session_cancel_markers WHERE session_id = $1 AND request_id = $2",
        session_id,
        request_id,
    )
    return _row_to_session_cancel_marker(row) if row is not None else None


async def list_unharvested_session_cancel_markers(
    conn: asyncpg.Connection[Any], session_id: str
) -> list[SessionCancelMarker]:
    """The session's cancel-markers the step has not yet applied (the C2 sweep's set)."""
    rows = await conn.fetch(
        "SELECT * FROM session_cancel_markers "
        "WHERE session_id = $1 AND harvested_at IS NULL ORDER BY created_at",
        session_id,
    )
    return [_row_to_session_cancel_marker(r) for r in rows]


async def mark_session_cancel_marker_harvested(
    conn: asyncpg.Connection[Any], *, session_id: str, request_id: str
) -> None:
    """Flip a marker to harvested once the session's step has applied it (idempotent)."""
    await conn.execute(
        "UPDATE session_cancel_markers SET harvested_at = now() "
        "WHERE session_id = $1 AND request_id = $2 AND harvested_at IS NULL",
        session_id,
        request_id,
    )


async def list_session_ids_with_unharvested_cancel_marker(
    conn: asyncpg.Connection[Any], *, session_id: str | None = None
) -> set[str]:
    """Non-archived session ids carrying an unharvested cancel-marker (the C2 sweep set).

    The session-side analog of the run sweep's unharvested-cancel-signal clause. Selected
    OUTSIDE the active/errored filter by its caller (``find_sessions_needing_inference``): a
    marked session must run its cancel leaf even when idle or errored-parked — it still owes
    a ``cancelled`` response. Optionally scoped to one ``session_id`` (the targeted sweep).
    """
    scope = "AND m.session_id = $1" if session_id else ""
    params: list[Any] = [session_id] if session_id else []
    rows = await conn.fetch(
        "SELECT DISTINCT m.session_id FROM session_cancel_markers m "
        "JOIN sessions s ON s.id = m.session_id AND s.archived_at IS NULL "
        f"WHERE m.harvested_at IS NULL {scope}",
        *params,
    )
    return {r["session_id"] for r in rows}
