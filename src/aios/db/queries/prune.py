"""Age-based prune for RECLAIMABLE instance ephemera (T6, aios#1461).

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.

This is the DB-side of the ratified T6 convention: **prune reclaimable instance
ephemera and unreferenced history past a retention window; NEVER prune anything
a live session pins or that constitutes institutional memory.** It refines, not
violates, never-delete.

Modeled on the two prunes that already exist in aios:

- ``prune_trigger_runs`` (``triggers.py``) — the audit-row prune. **Time-based
  by design**: a count-cap could evict a young ``run_completion`` claim row
  inside the dispatch-recovery horizon and re-arm a duplicate fire. Every prune
  here is likewise time-based (per the ``trigger_runs`` doctrine); there is no
  count-cap anywhere in this module.
- the archived-session sandbox reaper (``sandboxes.py:158``), age-keyed on
  ``archived_at < now() - make_interval(...)``.

Prune candidates (reclaimable):

- **terminal + archived runs** past ``wf_runs_retention_days`` — deleting the
  ``wf_runs`` row drops its ``WfRunEvent`` journal (the unbounded-growth driver)
  via the ``wf_run_events.run_id ... ON DELETE CASCADE`` FK (migration 0064). A
  run is only a candidate once ``archive_run`` (aios#9) has stamped
  ``archived_at`` AND its status is terminal — so a live/suspended run is never
  reached.
- **archived definitions** (agents / skills / workflows) past
  ``archived_definition_retention_days`` with **NO live session pinning** them —
  replay-stability requires the pinned version survive, so a definition any live
  (``archived_at IS NULL``) session still references is held.

SACRED — never pruned (the ratified set):

- memory content (``memory_stores`` / ``memories``) — never touched here,
- referenced session history (session event/message history referenced as
  memory) — never touched here,
- ``agent_versions`` / ``skill_versions`` / ``workflow_versions`` that a **live
  session still pins** — the live-pin predicate below holds the parent
  definition (and its version history) whenever a live session references it,
- accounts (purge stays a deliberate operator ceremony) — never touched here.

Every function is idempotent: re-running a sweep over an already-pruned window
deletes nothing further and returns 0. All are time-based (no count-cap).
"""

from __future__ import annotations

from typing import Any

import asyncpg


async def prune_archived_runs(
    conn: asyncpg.Connection[Any], *, retention_days: int, row_limit: int = 500
) -> int:
    """Delete bounded terminal child detail while retaining durable run summaries."""
    deleted = 0
    rows = await conn.fetch(
        """SELECT id FROM wf_runs
             WHERE archived_at IS NOT NULL
               AND archived_at < now() - make_interval(days => $1)
               AND terminal_summary IS NOT NULL
               AND journal_pruned_at IS NULL
               AND status IN ('completed','errored','cancelled')
             ORDER BY archived_at, id LIMIT $2""",
        retention_days,
        row_limit,
    )
    for row in rows:
        run_id = row["id"]
        for table, order in (("wf_run_events", "seq"), ("wf_run_signals", "delivered_at")):
            result = await conn.execute(
                f"""DELETE FROM {table} WHERE ctid IN (
                    SELECT child.ctid FROM {table} child JOIN wf_runs run ON run.id=child.run_id
                    WHERE child.run_id=$1 AND run.status IN ('completed','errored','cancelled')
                      AND run.terminal_summary IS NOT NULL
                      AND run.archived_at < now() - make_interval(days => $2)
                    ORDER BY child.{order} LIMIT $3)""",
                run_id,
                retention_days,
                row_limit,
            )
            deleted += int(result.split()[-1])
        remaining = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM wf_run_events WHERE run_id=$1) OR "
            "EXISTS(SELECT 1 FROM wf_run_signals WHERE run_id=$1)",
            run_id,
        )
        if not remaining:
            await conn.execute(
                "UPDATE wf_runs SET journal_pruned_at=now() WHERE id=$1 "
                "AND status IN ('completed','errored','cancelled')",
                run_id,
            )
    return deleted


async def reconcile_terminal_archival_batch(
    conn: asyncpg.Connection[Any], *, grace_days: int = 7, row_limit: int = 500
) -> int:
    """Archive a bounded batch of terminal runs older than the grace window.

    The terminal transition updates ``updated_at``, making it the age key for
    historical rows that predate automatic archival. Already-archived legacy
    rows missing their durable summary are also projected, irrespective of age.
    """
    result = await conn.execute(
        """WITH candidates AS (
               SELECT r.id, e.payload
                 FROM wf_runs r
                 LEFT JOIN LATERAL (
                     SELECT payload FROM wf_run_events
                      WHERE run_id = r.id AND type = 'run_completed'
                      ORDER BY seq DESC LIMIT 1
                 ) e ON true
                WHERE r.status IN ('completed','errored','cancelled')
                  AND ((r.archived_at IS NULL
                        AND r.updated_at < now() - make_interval(days => $1))
                       OR (r.archived_at IS NOT NULL AND r.terminal_summary IS NULL))
                ORDER BY r.updated_at, r.id
                LIMIT $2
           )
           UPDATE wf_runs r
              SET archived_at = COALESCE(r.archived_at, now()),
                  terminal_summary = COALESCE(r.terminal_summary,
                      jsonb_strip_nulls(jsonb_build_object(
                          'is_error', c.payload->'is_error',
                          'error', c.payload->'error',
                          'usage', c.payload->'usage',
                          'duration_ms', c.payload->'duration_ms',
                          'cancelled', (r.status = 'cancelled')
                      )))
             FROM candidates c WHERE r.id = c.id
               AND r.status IN ('completed','errored','cancelled')
               AND (r.archived_at IS NULL OR r.terminal_summary IS NULL)""",
        grace_days,
        row_limit,
    )
    return int(result.split()[-1])


async def prune_unpinned_archived_agents(
    conn: asyncpg.Connection[Any],
    *,
    retention_days: int,
) -> int:
    """Delete archived agents with NO session referencing them; returns count.

    A candidate is an ``agents`` row with ``archived_at`` set and older than the
    window for which **no live session** (``sessions.archived_at IS NULL``)
    still pins the ``agent_id`` — the replay-stability guarantee: a live-pinned
    version is SACRED, so the whole definition (and its ``agent_versions``
    history) is held while any live session points at it.

    The guard is widened to **any** session, not only live ones, for two
    independent reasons that both point the same way: (1) an archived session's
    event/message history can be referenced as memory, which is sacred — so its
    pinned definition must survive too; and (2) ``sessions.agent_id REFERENCES
    agents(id)`` carries no ``ON DELETE CASCADE`` (migration 0001), so deleting a
    still-referenced agent would FK-violate regardless. Holding on any session
    is therefore both the safe semantics and the FK-correct one.

    There is a THIRD non-cascade FK to ``agents(id)``:
    ``session_templates.agent_id NOT NULL REFERENCES agents(id)`` (migration 0027,
    no ``ON DELETE`` → NO ACTION). A ``session_templates`` row (a frozen recipe
    used for ``per_chat`` connector spawns) pins its agent regardless of either
    party's archive state, so deleting a template-pinned archived agent would
    ``ForeignKeyViolationError``. The same ``NOT EXISTS`` guard is therefore
    applied against ``session_templates`` as well — making the prune FK-correct
    on every non-cascade reference to ``agents(id)``.

    ``agent_versions REFERENCES agents(id)`` likewise has no cascade, so the
    version history of a now-unreferenced archived agent is deleted first, in the
    same transaction, before the parent row. Time-based, idempotent, unscoped.
    """
    async with conn.transaction():
        await conn.execute(
            """
            DELETE FROM agent_versions av
             USING agents a
             WHERE av.agent_id = a.id
               AND a.archived_at IS NOT NULL
               AND a.archived_at < now() - make_interval(days => $1)
               AND NOT EXISTS (
                   SELECT 1 FROM sessions s WHERE s.agent_id = a.id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM session_templates st WHERE st.agent_id = a.id
               )
            """,
            retention_days,
        )
        result = await conn.execute(
            """
            DELETE FROM agents a
             WHERE a.archived_at IS NOT NULL
               AND a.archived_at < now() - make_interval(days => $1)
               AND NOT EXISTS (
                   SELECT 1 FROM sessions s WHERE s.agent_id = a.id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM session_templates st WHERE st.agent_id = a.id
               )
            """,
            retention_days,
        )
    return int(result.split()[-1])


async def prune_unpinned_archived_workflows(
    conn: asyncpg.Connection[Any],
    *,
    retention_days: int,
) -> int:
    """Delete archived workflows with NO run pinning them; returns count.

    A candidate is a ``workflows`` row with ``archived_at`` set and older than
    the window for which **no live (non-archived) run** still pins the
    ``workflow_id`` — a live run pins the workflow's script/version for replay,
    so the definition (and ``workflow_versions`` history) is held while any live
    run points at it.

    The guard is widened to **any** run (not only live ones) because
    ``wf_runs.workflow_id REFERENCES workflows(id) ON DELETE CASCADE`` (migration
    0064): deleting a workflow would CASCADE-delete every run of it — including
    terminal+archived runs that ``prune_archived_runs`` reclaims on its own
    schedule, and any run not yet past its own retention window. Holding the
    workflow until ZERO runs reference it keeps the two prunes independent and
    never destroys a run (or its journal) out from under its own window.
    ``workflow_versions`` cascades from ``workflows`` on delete (migration 0112),
    so the version history goes with the now-unreferenced definition.

    Time-based, idempotent, unscoped across accounts.
    """
    result = await conn.execute(
        """
        DELETE FROM workflows w
         WHERE w.archived_at IS NOT NULL
           AND w.archived_at < now() - make_interval(days => $1)
           AND NOT EXISTS (
               SELECT 1 FROM wf_runs r WHERE r.workflow_id = w.id
           )
        """,
        retention_days,
    )
    return int(result.split()[-1])


async def prune_unpinned_archived_skills(
    conn: asyncpg.Connection[Any],
    *,
    retention_days: int,
) -> int:
    """Delete archived skills with NO live agent referencing them; count.

    A candidate is a ``skills`` row with ``archived_at`` set and older than the
    window for which **no live agent** (``agents.archived_at IS NULL``) still
    binds the ``skill_id`` in its ``skills`` JSONB reference list
    (``[{skill_id, version}]``, migration 0009). A live agent that binds a skill
    must be able to load it, so the skill definition (and ``skill_versions``
    history) is held while any live agent references it.

    ``skill_versions REFERENCES skills(id)`` carries no ``ON DELETE CASCADE``
    (migration 0009), so the version history of a now-unbound archived skill is
    deleted first, in the same transaction, before the parent row. Time-based,
    idempotent, unscoped across accounts.
    """
    # Reference predicate: no LIVE agent (current ``skills`` JSONB) binds it.
    # Archived agents' bindings are not consulted — an archived agent cannot be
    # loaded/run, so its stale binding does not pin a skill. (Unlike the agents
    # prune, there is no FK from agents → skills, so a leftover archived binding
    # cannot FK-violate; only a live binding is a real, loadable reference.)
    no_live_agent_binds = """
        NOT EXISTS (
            SELECT 1
              FROM agents a,
                   jsonb_array_elements(a.skills) AS ref
             WHERE a.archived_at IS NULL
               AND ref->>'skill_id' = sk.id
        )
    """
    async with conn.transaction():
        await conn.execute(
            f"""
            DELETE FROM skill_versions sv
             USING skills sk
             WHERE sv.skill_id = sk.id
               AND sk.archived_at IS NOT NULL
               AND sk.archived_at < now() - make_interval(days => $1)
               AND {no_live_agent_binds}
            """,
            retention_days,
        )
        result = await conn.execute(
            f"""
            DELETE FROM skills sk
             WHERE sk.archived_at IS NOT NULL
               AND sk.archived_at < now() - make_interval(days => $1)
               AND {no_live_agent_binds}
            """,
            retention_days,
        )
    return int(result.split()[-1])
