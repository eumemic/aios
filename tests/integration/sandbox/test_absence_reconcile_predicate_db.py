"""DB-backed contract for the destructive reconcile-by-absence predicate.

``unscoped_reconcile_absent_host_snapshots`` is the ONE statement in this PR
that destroys a session's DB linkage to its durable filesystem. Every guard in
it is a never-delete protection, and until now no test executed the UPDATE at
all: deleting the ``AND id <> ALL($4::text[])`` (liveness) conjunct left the
whole suite green while the statement cleared a LIVE session's pointer.

This test drives the *production* query function against real Postgres — not a
transcribed copy of the SQL — so that mutating any conjunct in
``src/aios/db/queries/sandboxes.py`` is what turns it red.

Six rows, one per branch of the predicate:

* ``sess_dead``      — host-owned, absent, old, unprotected  ⇒ the ONLY clear.
* ``sess_live``      — identical to ``sess_dead`` except this worker holds a
                       handle for it (``protected_session_ids``). Guards the
                       ``id <> ALL($4)`` conjunct.
* ``sess_young``     — absent but written inside the recency floor. Guards
                       ``snapshot_updated_at <= now() - $5``.
* ``sess_racing``    — absent and old, but written AFTER the enumeration
                       started. Guards the ``<= $3`` observed-before CAS.
* ``sess_present``   — its ref appears in the enumeration. Guards
                       ``snapshot_ref <> ALL($2)``.
* ``sess_nulltime``  — pointer with a NULL ``snapshot_updated_at``. Both
                       timestamp comparisons are NULL ⇒ never eligible; the row
                       must survive rather than be swept by a three-valued-logic
                       accident.
* ``sess_otherhost`` — owned by a different ``snapshot_host``. Guards
                       ``snapshot_host = $1``: one host's enumeration is
                       evidence about NO other host's disk.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from aios.db import queries

pytestmark = pytest.mark.integration

_HOST = "inst_gc_host"
_OTHER_HOST = "inst_gc_other"


async def _seed(conn: asyncpg.Connection[Any], now: datetime) -> None:
    await conn.execute(
        "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
        "VALUES ('acc_absence', NULL, TRUE, 'absence')"
    )
    await conn.execute(
        "INSERT INTO environments (id, account_id, name) "
        "VALUES ('env_absence', 'acc_absence', 'env')"
    )
    await conn.execute(
        "INSERT INTO agents (id, account_id, name, model, system, version) "
        "VALUES ('agent_absence', 'acc_absence', 'agent', 'test/model', '', 1)"
    )
    # (session_id, snapshot_ref, snapshot_host, snapshot_updated_at)
    rows: list[tuple[str, str, str, datetime | None]] = [
        # Stale: host-owned, absent from the enumeration, comfortably older
        # than both the CAS bound and the recency floor.
        ("sess_dead", "snap:dead", _HOST, now - timedelta(hours=6)),
        # Identical to sess_dead in every column — only the caller-supplied
        # liveness set distinguishes it.
        ("sess_live", "snap:live", _HOST, now - timedelta(hours=6)),
        # Absent, but written 1 minute ago: inside the 15-minute recency
        # floor. Isolated by the third test, whose observed_before is "now"
        # so the CAS admits this row and only the floor can refuse it.
        ("sess_young", "snap:young", _HOST, now - timedelta(minutes=1)),
        # Absent and 30 minutes old — past the 15-minute floor — but written
        # AFTER the enumeration began in test one (observed_before = 40 min
        # ago), so only the CAS can refuse it there.
        ("sess_racing", "snap:racing", _HOST, now - timedelta(minutes=30)),
        # Present in the enumeration.
        ("sess_present", "snap:present", _HOST, now - timedelta(hours=6)),
        # NULL timestamp: no evidence of age at all.
        ("sess_nulltime", "snap:nulltime", _HOST, None),
        # Another host's pointer.
        ("sess_otherhost", "snap:otherhost", _OTHER_HOST, now - timedelta(hours=6)),
    ]
    for sid, ref, host, updated in rows:
        await conn.execute(
            """
            INSERT INTO sessions
                (id, agent_id, environment_id, agent_version, title, metadata,
                 workspace_volume_path, env, account_id, last_event_seq,
                 snapshot_ref, snapshot_host, snapshot_bytes, snapshot_updated_at)
            VALUES ($1, 'agent_absence', 'env_absence', 1, 'session', '{}'::jsonb,
                    '/tmp/workspace', '{}'::jsonb, 'acc_absence', 0,
                    $2, $3, 1024, $4)
            """,
            sid,
            ref,
            host,
            updated,
        )


async def _cleared_session_ids(conn: asyncpg.Connection[Any]) -> list[str]:
    """Session ids whose durable-FS pointer is now NULL, sorted."""
    rows = await conn.fetch(
        "SELECT id FROM sessions WHERE snapshot_ref IS NULL ORDER BY id",
    )
    return [row["id"] for row in rows]


async def test_only_the_stale_unprotected_host_pointer_is_cleared(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """Exactly one of seven pointers is eligible; the other six are protected.

    ``sess_live`` is the mutation target: dropping ``AND id <> ALL($4::text[])``
    from the production statement makes this assertion read
    ``['sess_dead', 'sess_live']`` — a live session's durable-FS pointer
    destroyed on nothing more than a tag failing to appear in one enumeration.
    """
    conn: asyncpg.Connection[Any] = await asyncpg.connect(migrated_db_url)
    try:
        now = datetime.now(UTC)
        await _seed(conn, now)

        cleared = await queries.unscoped_reconcile_absent_host_snapshots(
            conn,
            _HOST,
            ["snap:present"],
            # Enumeration began 40 minutes ago, so sess_racing (written 30
            # minutes ago, i.e. AFTER it) is past the recency floor and is
            # refused by the CAS conjunct alone.
            observed_before=now - timedelta(minutes=40),
            protected_session_ids=["sess_live"],
            min_age=timedelta(minutes=15),
        )

        assert await _cleared_session_ids(conn) == ["sess_dead"]
        assert cleared == 1

        # Every other pointer is intact — not merely "not NULL", but still
        # naming the same artifact on the same host.
        survivors = await conn.fetch(
            "SELECT id, snapshot_ref, snapshot_host, snapshot_bytes FROM sessions "
            "WHERE id <> 'sess_dead' ORDER BY id"
        )
        assert [
            (r["id"], r["snapshot_ref"], r["snapshot_host"], r["snapshot_bytes"]) for r in survivors
        ] == [
            ("sess_live", "snap:live", _HOST, 1024),
            ("sess_nulltime", "snap:nulltime", _HOST, 1024),
            ("sess_otherhost", "snap:otherhost", _OTHER_HOST, 1024),
            ("sess_present", "snap:present", _HOST, 1024),
            ("sess_racing", "snap:racing", _HOST, 1024),
            ("sess_young", "snap:young", _HOST, 1024),
        ]
    finally:
        await conn.close()


async def test_empty_protection_sets_still_reconcile_the_stale_pointers(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """NEGATIVE CONTROL: the guards do not wedge collection shut.

    A guard only ever observed refusing is indistinguishable from one that
    refuses everything, and a GC that declines every tick trades "destroys live
    state" for "never collects". With no live handles and nothing enumerated,
    every genuinely-collectible host pointer IS reconciled.

    ``observed_before=now`` also isolates the recency floor: it is the only
    conjunct that can refuse ``sess_young`` here (the CAS admits it), so
    deleting ``AND snapshot_updated_at <= now() - $5::interval`` turns this red.
    """
    conn: asyncpg.Connection[Any] = await asyncpg.connect(migrated_db_url)
    try:
        now = datetime.now(UTC)
        await _seed(conn, now)

        cleared = await queries.unscoped_reconcile_absent_host_snapshots(
            conn,
            _HOST,
            [],  # nothing present on the host
            observed_before=now,  # enumeration started just now
            protected_session_ids=[],  # this worker holds no handles
            min_age=timedelta(minutes=15),
        )

        assert await _cleared_session_ids(conn) == [
            "sess_dead",
            "sess_live",
            "sess_present",
            "sess_racing",
        ]
        assert cleared == 4
        # Still refused: inside the floor, NULL-timestamped, or another host's.
        survivors = await conn.fetch(
            "SELECT id FROM sessions WHERE snapshot_ref IS NOT NULL ORDER BY id"
        )
        assert [r["id"] for r in survivors] == ["sess_nulltime", "sess_otherhost", "sess_young"]
    finally:
        await conn.close()


async def test_absence_reconcile_makes_stranded_reset_notice_deliverable(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """Crash-recovery: a stranded pending reset notice becomes deliverable.

    Pressure reclamation records its reset intent
    (``snapshot_reset_pending_reason``) BEFORE removing the image, and only
    marks it deliverable (``snapshot_reset_pending_ready = TRUE``) in the same
    statement that clears the exact-ref pointer
    (``unscoped_compare_and_clear_session_snapshot``). If the process crashes
    after the image is physically gone but before that clear, the intent is
    stranded: pointer still set, marker unready, so
    ``unscoped_list_pending_snapshot_reset_notices`` excludes it and the
    filesystem-reset lifecycle event is never emitted.

    Absence reconcile is precisely the pass that later proves the image gone.
    It must clear the pointer AND atomically flip the stranded marker ready.
    Deleting the ``snapshot_reset_pending_ready = CASE ... END`` clause from the
    production UPDATE makes the final ``pending`` assertion read ``[]`` — the
    notice lost forever.
    """
    conn: asyncpg.Connection[Any] = await asyncpg.connect(migrated_db_url)
    try:
        now = datetime.now(UTC)
        await _seed(conn, now)

        # sess_dead already exists (host-owned, absent, old). Strand a pending
        # reset intent on it that never became ready — the exact crash window.
        await conn.execute(
            """UPDATE sessions
                  SET snapshot_reset_pending_reason = 'snapshot_pool_pressure',
                      snapshot_reset_pending_ready = FALSE
                WHERE id = 'sess_dead'"""
        )
        # Precondition: the stranded notice is NOT deliverable yet.
        assert await queries.unscoped_list_pending_snapshot_reset_notices(conn) == []

        cleared = await queries.unscoped_reconcile_absent_host_snapshots(
            conn,
            _HOST,
            ["snap:present"],  # sess_dead's ref is gone
            observed_before=now - timedelta(minutes=40),
            protected_session_ids=["sess_live"],
            min_age=timedelta(minutes=15),
        )
        assert cleared == 1

        # The pointer is cleared AND the stranded notice is now deliverable.
        row = await conn.fetchrow(
            "SELECT snapshot_ref, snapshot_reset_pending_reason, "
            "snapshot_reset_pending_ready FROM sessions WHERE id = 'sess_dead'"
        )
        assert row["snapshot_ref"] is None
        assert row["snapshot_reset_pending_reason"] == "snapshot_pool_pressure"
        assert row["snapshot_reset_pending_ready"] is True
        assert await queries.unscoped_list_pending_snapshot_reset_notices(conn) == [
            ("sess_dead", "snapshot_pool_pressure")
        ]
    finally:
        await conn.close()


async def test_absence_reconcile_never_fabricates_a_reset_notice(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """OVER-CORRECTION GUARD: a cleared pointer with no intent stays quiet.

    The recovery only rescues rows that ALREADY carry a pending reset reason.
    The degenerate over-correction — unconditionally setting
    ``snapshot_reset_pending_ready = TRUE`` on every reconcile — would emit a
    filesystem-reset lifecycle event for ordinary GC of a session that never
    had a pressure reclamation. ``sess_dead`` here has no reset intent, so after
    reconcile it must have NO deliverable notice.

    Replacing the CASE with a bare ``snapshot_reset_pending_ready = TRUE`` turns
    this red: ``pending`` would list ``('sess_dead', NULL)`` (or the list would
    be non-empty), fabricating a reset that never happened.
    """
    conn: asyncpg.Connection[Any] = await asyncpg.connect(migrated_db_url)
    try:
        now = datetime.now(UTC)
        await _seed(conn, now)
        # sess_dead has NO pending reset intent (seeded without one).

        cleared = await queries.unscoped_reconcile_absent_host_snapshots(
            conn,
            _HOST,
            ["snap:present"],
            observed_before=now - timedelta(minutes=40),
            protected_session_ids=["sess_live"],
            min_age=timedelta(minutes=15),
        )
        assert cleared == 1

        row = await conn.fetchrow(
            "SELECT snapshot_ref, snapshot_reset_pending_reason, "
            "snapshot_reset_pending_ready FROM sessions WHERE id = 'sess_dead'"
        )
        assert row["snapshot_ref"] is None
        assert row["snapshot_reset_pending_reason"] is None
        assert row["snapshot_reset_pending_ready"] is False
        # POSITIVE CONTROL side: absolutely no notice fabricated.
        assert await queries.unscoped_list_pending_snapshot_reset_notices(conn) == []
    finally:
        await conn.close()
