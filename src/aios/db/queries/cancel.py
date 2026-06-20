"""Cancel-supervision side-table queries (cancel-design §0/§9).

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared contract. These are the durable primitives the recursive ``cancel_invocation``
cascade is built on: the ``cancel_intents`` tombstone (+ its §9 monotone quiescence
counter) and the session-side ``session_cancel_markers`` exit-marker (the run side
reuses ``wf_run_signals kind='cancel'`` verbatim). Every write is to a **side-table**,
never to a node's event log/journal — the single-writer invariant: a node's own step
under its own lock is the only writer of its log.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.models.invocations import CancelIntent, SessionCancelMarker


def _row_to_cancel_intent(row: asyncpg.Record) -> CancelIntent:
    return CancelIntent(
        servicer_kind=row["servicer_kind"],
        servicer_id=row["servicer_id"],
        request_id=row["request_id"],
        account_id=row["account_id"],
        outstanding=row["outstanding"],
        quiesced_at=row["quiesced_at"],
        created_at=row["created_at"],
    )


def _row_to_session_cancel_marker(row: asyncpg.Record) -> SessionCancelMarker:
    return SessionCancelMarker(
        session_id=row["session_id"],
        request_id=row["request_id"],
        account_id=row["account_id"],
        harvested_at=row["harvested_at"],
        created_at=row["created_at"],
    )


# ─── cancel_intents tombstone ────────────────────────────────────────────────


async def insert_cancel_intent(
    conn: asyncpg.Connection[Any],
    *,
    servicer_kind: str,
    servicer_id: str,
    request_id: str,
    account_id: str,
) -> CancelIntent:
    """Write (idempotently) the cancel tombstone for an edge handle, returning it.

    A repeated ``cancel_invocation`` of the same edge is a no-op: the no-op
    ``DO UPDATE`` matches the existing row so ``RETURNING`` yields it in one round-trip
    WITHOUT resetting its ``outstanding`` counter (first-cancel wins).
    """
    row = await conn.fetchrow(
        """
        INSERT INTO cancel_intents (servicer_kind, servicer_id, request_id, account_id)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (servicer_kind, servicer_id, request_id)
            DO UPDATE SET created_at = cancel_intents.created_at
        RETURNING *
        """,
        servicer_kind,
        servicer_id,
        request_id,
        account_id,
    )
    assert row is not None
    return _row_to_cancel_intent(row)


async def get_cancel_intent(
    conn: asyncpg.Connection[Any], *, servicer_kind: str, servicer_id: str, request_id: str
) -> CancelIntent | None:
    row = await conn.fetchrow(
        "SELECT * FROM cancel_intents "
        "WHERE servicer_kind = $1 AND servicer_id = $2 AND request_id = $3",
        servicer_kind,
        servicer_id,
        request_id,
    )
    return _row_to_cancel_intent(row) if row is not None else None


async def adjust_cancel_outstanding(
    conn: asyncpg.Connection[Any],
    *,
    servicer_kind: str,
    servicer_id: str,
    request_id: str,
    delta: int,
) -> CancelIntent | None:
    """Apply the §9 quiescence ``delta`` to the tombstone's monotone counter.

    Returns the updated row (or ``None`` if the tombstone is gone). ``quiesced_at``
    latches the first time ``outstanding`` reaches 0 and never un-sets (monotone) — the
    writer that drives it to 0 stamps the cascade's completion. Each node calls this in
    its own terminal/withdraw transaction with ``delta = (#children marked) - 1``.
    """
    row = await conn.fetchrow(
        """
        UPDATE cancel_intents
        SET outstanding = outstanding + $4,
            quiesced_at = CASE
                WHEN quiesced_at IS NULL AND outstanding + $4 <= 0 THEN now()
                ELSE quiesced_at
            END
        WHERE servicer_kind = $1 AND servicer_id = $2 AND request_id = $3
        RETURNING *
        """,
        servicer_kind,
        servicer_id,
        request_id,
        delta,
    )
    return _row_to_cancel_intent(row) if row is not None else None


# ─── session_cancel_markers exit-marker ──────────────────────────────────────


async def insert_session_cancel_marker(
    conn: asyncpg.Connection[Any], *, session_id: str, request_id: str, account_id: str
) -> bool:
    """Seed a cancel-marker on a session edge; ``True`` iff newly inserted.

    ``ON CONFLICT DO NOTHING`` is the cascade's idempotency (re-propagation / a repeated
    cancel is a no-op): ``RETURNING`` yields the row only on a fresh insert, so the bool
    lets the caller account the §9 counter exactly once per genuinely-new mark.
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
