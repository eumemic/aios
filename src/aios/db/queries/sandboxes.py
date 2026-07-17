"""Durable session-sandbox snapshot-pointer queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import asyncpg

from aios.db.queries.sessions import session_active_predicate

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


async def unscoped_live_session_ids(
    conn: asyncpg.Connection[Any], session_ids: Sequence[str]
) -> set[str]:
    """Return the subset of ``session_ids`` that are DB-live (#1192).

    DB-live = the session row exists and is NOT soft-archived
    (``archived_at IS NULL``). This is the keep-set for the
    ``_session_repos`` host scratch reaper: a session container released by
    the idle reaper (container-absent) is still live in the DB, and its
    per-session working-tree clones must be preserved.

    The clones are *reconstructible* (github-clone rmtree+re-clones on the
    next provision), so an archived/deleted session being absent here — and
    hence its clones reaped — is at worst a re-clone on a (rare) unarchive,
    never data loss. A session id absent from this set means "not live ⇒
    eligible to reap".

    Worker-side and unscoped (per the ``unscoped_`` convention): the reaper
    holds only the session ids it scraped off disk, across all accounts.
    """
    if not session_ids:
        return set()
    rows = await conn.fetch(
        """
        SELECT id FROM sessions
         WHERE id = ANY($1::text[]) AND archived_at IS NULL
        """,
        list(session_ids),
    )
    return {row["id"] for row in rows}


async def unscoped_reapable_archived_workspaces(
    conn: asyncpg.Connection[Any], *, min_archived_age_seconds: float
) -> list[asyncpg.Record]:
    """Return ``(id, account_id, workspace_volume_path)`` for sessions whose
    ``/workspace`` host dir is reap-eligible — the archived-session workspace
    reaper (aios#40, the "45G hole").

    A session qualifies iff ALL hold:

    * ``archived_at IS NOT NULL`` — archived-only. Session archival is terminal
      and permanent: no code path ever clears ``sessions.archived_at`` (the
      archive fence at ``append_event`` rejects every later write), so an
      archived session never wakes, never re-provisions, never re-reads its
      ``/workspace``. We reap only archived sessions; a live/idle/running
      session is structurally excluded by this clause.

    * ``archived_at < now() - min_archived_age_seconds`` — the min-age floor, on
      *DB archive time* (not file mtime). Guards a just-archived session whose
      workspace something might still be reading in the seconds after archive.

    * ``NOT (active)`` — defense-in-depth. Archived dominates the read-path
      status label, but the explicit ``archive_session`` path carries no
      active-guard, so archived ∧ event-watermark-active is reachable for a
      step that was in flight at archive time. We re-confirm not-active here
      with the SAME predicate the wake sweep uses, read fresh at sweep time, so
      a session with work still pending is never a candidate.

    Returns the *stored* ``workspace_volume_path`` verbatim; the caller is
    responsible for the canonical-path confinement check (it reaps ONLY the
    derived ``workspace_root/<account_id>/<session_id>`` and only when the
    stored value resolves equal to it — a user-overridden / clone-shared /
    aliased path is therefore never reaped). Worker-side / unscoped: the reaper
    holds these rows across all accounts.
    """
    rows: list[asyncpg.Record] = await conn.fetch(
        f"""
        SELECT id, account_id, workspace_volume_path
          FROM sessions
         WHERE archived_at IS NOT NULL
           AND archived_at < now() - make_interval(secs => $1::double precision)
           AND NOT {session_active_predicate("sessions")}
        """,
        float(min_archived_age_seconds),
    )
    return rows


async def unscoped_live_workspace_volume_paths(
    conn: asyncpg.Connection[Any],
) -> list[str]:
    """Return paths currently borrowed by live sessions or shared workflow runs.

    The keep-set includes every non-archived session's ``workspace_volume_path``
    and every non-terminal shared ``wf_runs.workspace_path``. A shared workflow
    run keeps executing against its launcher's session workspace even after the
    launcher is archived, so omitting those run pointers can delete a live run's
    filesystem.

    The keep-set for the archived-workspace reaper's live-clone cross-check
    (aios#40, same never-delete class as the confinement gate). ``clone_session``
    lets a live clone *share* the volume of another session — and a live clone
    can legitimately point at an ARCHIVED parent's OWN canonical default path
    (``<root>/<account>/<parent>``). Reaping that archived parent's row would
    ``rmtree`` the very directory the live clone is using ⇒ cross-session live
    data loss.

    Returns the *stored* paths verbatim, across all accounts (``unscoped_``); the
    caller realpath-normalizes them into a keep-set and skips any reap candidate
    whose canonical realpath collides with a live path. NULL/empty stored values
    are filtered out (they can never realpath-collide with a real canonical dir).
    """
    rows = await conn.fetch(
        """
        SELECT workspace_volume_path
          FROM sessions
         WHERE archived_at IS NULL
           AND workspace_volume_path IS NOT NULL
           AND workspace_volume_path <> ''
        UNION
        SELECT workspace_path AS workspace_volume_path
          FROM wf_runs
         WHERE workspace_mode = 'shared'
           AND status IN ('pending', 'running', 'suspended')
           AND workspace_path IS NOT NULL
           AND workspace_path <> ''
        """,
    )
    return [row["workspace_volume_path"] for row in rows]


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
