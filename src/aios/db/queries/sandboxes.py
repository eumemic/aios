"""Durable session-sandbox snapshot-pointer queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import asyncpg

# ─── durable session sandboxes: snapshot pointer (§5.1) ───────────────────────
#
# Worker-side, unscoped (per the ``unscoped_`` convention): the snapshot
# lifecycle runs entirely worker-internal, keyed only by the session_id the
# worker already holds. The pointer is written inside the snapshot critical
# section (after commit success, before ``rm``) and reconciled by the GC tick.
# v1 is single-host, so writes are direct; the §5.5 multi-host compare-and-swap
# and ownership-gating refinements are deferred (the ``snapshot_host`` column
# is the seam that makes them additive).


async def unscoped_set_session_snapshot(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    ref: str,
    host: str,
    snapshot_bytes: int,
) -> None:
    """Point a session at a just-committed snapshot. ``snapshot_updated_at`` = now()."""
    await conn.execute(
        """
        UPDATE sessions
           SET snapshot_ref = $2,
               snapshot_host = $3,
               snapshot_bytes = $4,
               snapshot_updated_at = now()
         WHERE id = $1
        """,
        session_id,
        ref,
        host,
        snapshot_bytes,
    )


async def unscoped_get_session_snapshot_bytes(
    conn: asyncpg.Connection[Any], session_id: str
) -> int | None:
    """The session's last-recorded ``snapshot_bytes`` (``None`` if no snapshot).

    Read by the release path to edge-trigger the ``sandbox_fs_over_limit``
    notice only on the crossing (previous ≤ limit < new), not every cycle.
    """
    row = await conn.fetchrow("SELECT snapshot_bytes FROM sessions WHERE id = $1", session_id)
    if row is None:
        return None
    value = row["snapshot_bytes"]
    return int(value) if value is not None else None


async def unscoped_clear_session_snapshot(conn: asyncpg.Connection[Any], session_id: str) -> None:
    """Clear a session's snapshot pointer (all four columns NULL).

    Used on a detected reset (snapshot-missing / base-image drift) and by the
    GC pass-4 reconcile when a canonical artifact is removed.
    """
    await conn.execute(
        """
        UPDATE sessions
           SET snapshot_ref = NULL,
               snapshot_host = NULL,
               snapshot_bytes = NULL,
               snapshot_updated_at = now()
         WHERE id = $1
        """,
        session_id,
    )


async def gc_snapshot_session_states(
    conn: asyncpg.Connection[Any], session_ids: Sequence[str]
) -> list[asyncpg.Record]:
    """Batch-load the GC's per-session decision inputs for ``session_ids``.

    Returns one row per *existing* session (a deleted session is simply
    absent — its snapshot is collectible). Columns:

    * ``id`` / ``account_id`` / ``archived_at``
    * ``last_event_at`` — the dormancy probe: the ``created_at`` of the event
      at ``seq == sessions.last_event_seq``. A **point probe** on the
      ``(session_id, seq)`` primary key (NOT ``MAX(events.created_at)``, which
      has no index), so the retain rule stays cheap over arbitrarily large
      sessions.
    * ``snapshot_ref`` / ``snapshot_host`` / ``snapshot_bytes`` — the pointer,
      for the ownership-gated pass-4 reconcile.
    """
    if not session_ids:
        return []
    rows: list[asyncpg.Record] = await conn.fetch(
        """
        SELECT s.id,
               s.account_id,
               s.archived_at,
               s.snapshot_ref,
               s.snapshot_host,
               s.snapshot_bytes,
               (SELECT e.created_at FROM events e
                 WHERE e.session_id = s.id AND e.seq = s.last_event_seq) AS last_event_at
          FROM sessions s
         WHERE s.id = ANY($1::text[])
        """,
        list(session_ids),
    )
    return rows
