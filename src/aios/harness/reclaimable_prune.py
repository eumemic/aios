"""Periodic prune sweep for RECLAIMABLE instance ephemera (T6, aios#1461).

The harness-side driver for the age-based prune in
``aios.db.queries.prune``. Runs from a periodic maintenance sweep (an immediate
first sweep at worker startup, then every
``reclaimable_prune_interval_seconds``), mirroring the host scratch-dir reaper
and the ``trigger_runner`` maintenance pair.

What it reclaims (all time-based, per the ``trigger_runs`` doctrine — never a
count-cap):

- terminal+archived ``wf_runs`` past ``wf_runs_retention_days`` (dropping their
  ``WfRunEvent`` journals via ``ON DELETE CASCADE``),
- archived agent/skill/workflow definitions past
  ``archived_definition_retention_days`` that NO live session/run still pins.

What it NEVER touches (the ratified sacred set): memory content
(``memory_stores`` / ``memories``), referenced session history, any version a
live session pins, and accounts. The sweep is idempotent and safe to run
repeatedly; it logs what it reclaimed. Honours the ``reclaimable_prune_enabled``
kill-switch so a worker mid disk-incident can disable DB-row deletion without a
redeploy.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.logging import get_logger

log = get_logger("aios.harness.reclaimable_prune")


class PruneResult(NamedTuple):
    """Per-sweep tally of reclaimed rows (0s when the kill-switch is off)."""

    runs: int
    agents: int
    workflows: int
    skills: int

    @property
    def total(self) -> int:
        return self.runs + self.agents + self.workflows + self.skills


async def _prune_one_family(
    pool: asyncpg.Pool[Any],
    *,
    family: str,
    prune: Callable[[asyncpg.Connection[Any]], Awaitable[int]],
) -> int:
    """Run one family's prune in its OWN connection + try/except; 0 on failure.

    Each family is isolated so one family's failure cannot silently disable the
    others (the silent-failure mode the sweep is designed against): a raised
    ``ForeignKeyViolationError`` — e.g. a not-yet-guarded reference pinning an
    archived definition — must not skip the families that follow it. We log the
    per-family failure at ``exception`` level (so it is visible, never swallowed
    into a bare ``tick_failed``) and continue the sweep, returning 0 for that
    family this tick. The next sweep retries it.

    A fresh connection per family means a family's aborted transaction can never
    poison a sibling's connection state.
    """
    try:
        async with pool.acquire() as conn:
            return await prune(conn)
    except Exception:
        log.exception("reclaimable_prune.family_failed", family=family)
        return 0


async def sweep_reclaimable_ephemera(pool: asyncpg.Pool[Any]) -> PruneResult:
    """Run one prune sweep over all reclaimable families; return the tally.

    Honours the ``reclaimable_prune_enabled`` kill-switch (returns an all-zero
    result, deleting nothing, when off). Each family is pruned in its own
    connection and its own ``try/except``, so a failure on one family (a raise or
    a transient error) neither rolls back another's reclaim NOR aborts the rest
    of the sweep — the failing family logs and is skipped this tick (counted 0),
    the others still run. Idempotent: a second sweep over an already-pruned
    window deletes nothing further.
    """
    settings = get_settings()
    if not settings.reclaimable_prune_enabled:
        return PruneResult(0, 0, 0, 0)

    run_days = settings.wf_runs_retention_days
    def_days = settings.archived_definition_retention_days

    # Runs first: drains terminal+archived runs (and their journals) so the
    # subsequent workflow prune can reclaim a now-run-free archived workflow
    # within the same sweep where ages line up. Order is not required for
    # correctness (each prune re-reads liveness), only for promptness. Each
    # family is independently isolated, so a failure mid-sweep does not skip the
    # families that follow.
    runs = await _prune_one_family(
        pool,
        family="runs",
        prune=lambda c: queries.prune_archived_runs(c, retention_days=run_days),
    )
    agents = await _prune_one_family(
        pool,
        family="agents",
        prune=lambda c: queries.prune_unpinned_archived_agents(c, retention_days=def_days),
    )
    workflows = await _prune_one_family(
        pool,
        family="workflows",
        prune=lambda c: queries.prune_unpinned_archived_workflows(c, retention_days=def_days),
    )
    skills = await _prune_one_family(
        pool,
        family="skills",
        prune=lambda c: queries.prune_unpinned_archived_skills(c, retention_days=def_days),
    )

    result = PruneResult(runs=runs, agents=agents, workflows=workflows, skills=skills)
    if result.total:
        log.info(
            "reclaimable_prune.swept",
            runs=result.runs,
            agents=result.agents,
            workflows=result.workflows,
            skills=result.skills,
            wf_runs_retention_days=run_days,
            archived_definition_retention_days=def_days,
        )
    return result
