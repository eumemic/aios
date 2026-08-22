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
    """Per-sweep tally and any families whose result could not be determined."""

    runs: int
    agents: int
    workflows: int
    skills: int
    failed_families: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return self.runs + self.agents + self.workflows + self.skills

    @property
    def degraded(self) -> bool:
        """Whether at least one family failed, making the tally incomplete."""
        return bool(self.failed_families)


async def _prune_one_family(
    pool: asyncpg.Pool[Any],
    *,
    family: str,
    prune: Callable[[asyncpg.Connection[Any]], Awaitable[int]],
) -> tuple[int, bool]:
    """Run one family's prune in its OWN connection; return count and success.

    Each family is isolated so one family's failure cannot silently disable the
    others (the silent-failure mode the sweep is designed against): a raised
    ``ForeignKeyViolationError`` — e.g. a not-yet-guarded reference pinning an
    archived definition — must not skip the families that follow it. We log the
    per-family failure at ``exception`` level (so it is visible, never swallowed
    into a bare ``tick_failed``) and continue the sweep, returning a zero count
    plus a failed status for that family. The next sweep retries it.

    A fresh connection per family means a family's aborted transaction can never
    poison a sibling's connection state.
    """
    try:
        async with pool.acquire() as conn:
            return await prune(conn), True
    except Exception:
        log.exception("reclaimable_prune.family_failed", family=family)
        return 0, False


async def sweep_reclaimable_ephemera(pool: asyncpg.Pool[Any]) -> PruneResult:
    """Run one prune sweep over all reclaimable families; return the tally.

    Honours the ``reclaimable_prune_enabled`` kill-switch (returns an all-zero
    result, deleting nothing, when off). Each family is pruned in its own
    connection and its own ``try/except``, so a failure on one family (a raise or
    a transient error) neither rolls back another's reclaim NOR aborts the rest
    of the sweep — the failing family logs and is skipped this tick (counted 0),
    the others still run, and ``failed_families``/``degraded`` marks the returned
    tally as incomplete. Idempotent: a second sweep over an already-pruned window
    deletes nothing further.
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
    failed_families: list[str] = []

    _, archival_ok = await _prune_one_family(
        pool,
        family="runs_archival",
        prune=lambda c: queries.reconcile_terminal_archival_batch(
            c,
            grace_days=settings.wf_runs_archive_grace_days,
            row_limit=settings.reclaimable_prune_batch_rows,
        ),
    )
    if not archival_ok:
        failed_families.append("runs_archival")

    runs, runs_ok = await _prune_one_family(
        pool,
        family="runs",
        prune=lambda c: queries.prune_archived_runs(
            c,
            retention_days=run_days,
            row_limit=settings.reclaimable_prune_batch_rows,
        ),
    )
    if not runs_ok:
        failed_families.append("runs")

    agents, agents_ok = await _prune_one_family(
        pool,
        family="agents",
        prune=lambda c: queries.prune_unpinned_archived_agents(c, retention_days=def_days),
    )
    if not agents_ok:
        failed_families.append("agents")

    workflows, workflows_ok = await _prune_one_family(
        pool,
        family="workflows",
        prune=lambda c: queries.prune_unpinned_archived_workflows(c, retention_days=def_days),
    )
    if not workflows_ok:
        failed_families.append("workflows")

    skills, skills_ok = await _prune_one_family(
        pool,
        family="skills",
        prune=lambda c: queries.prune_unpinned_archived_skills(c, retention_days=def_days),
    )
    if not skills_ok:
        failed_families.append("skills")

    result = PruneResult(
        runs=runs,
        agents=agents,
        workflows=workflows,
        skills=skills,
        failed_families=tuple(failed_families),
    )
    # Emit degraded zero-count sweeps too: otherwise an all-family failure is
    # represented in the return value but disappears from the aggregate event.
    if result.total or result.degraded:
        log.info(
            "reclaimable_prune.swept",
            runs=result.runs,
            agents=result.agents,
            workflows=result.workflows,
            skills=result.skills,
            degraded=result.degraded,
            failed_families=result.failed_families,
            wf_runs_retention_days=run_days,
            archived_definition_retention_days=def_days,
        )
    return result
