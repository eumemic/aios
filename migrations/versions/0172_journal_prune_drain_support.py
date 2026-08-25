"""Journal-prune drain support: feeder backlog index + journal-table autovacuum tuning.

Revision ID: 0172
Revises: 0169
(Numbered 0172 because 0170/0171 are claimed by in-flight PRs #2202/#2247;
renumber at rebase if they land first — see #2212 for the fork-detection gap.)

Two schema-only changes for the #2245 drain (the 97 GB of terminal-run
journals that becomes prune-eligible as runs age past the retention window):

1. ``wf_runs_archival_backlog_idx`` — ``reconcile_terminal_archival_batch``
   (db/queries/prune.py) runs hourly forever, and its WHERE has no usable
   index: ``wf_runs_active_idx`` covers only NON-terminal rows (the exact
   complement), so every sweep seq-scans wf_runs. The predicate here is
   deliberately ``(archived_at IS NULL OR terminal_summary IS NULL)`` — BOTH
   arms of the feeder's OR must logically imply the index predicate or the
   planner cannot use it at all (verified empirically: the narrower
   ``archived_at IS NULL``-only form plans as a seq scan even with
   enable_seqscan=off; the widened form gives an ordered index scan matching
   the query's ``ORDER BY updated_at, id`` with no Sort node). Rows leave the
   index once the feeder stamps archived_at + terminal_summary, so it is
   near-empty in steady state.

   The ``DROP INDEX CONCURRENTLY IF EXISTS`` before the create is the retry
   path, not decoration: a failed CONCURRENTLY build (statement cancel,
   deadlock, dropped connection) leaves an INVALID index behind, and a bare
   ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` retry then reports "already
   exists, skipping" while the corpse is maintained on every write and never
   used by the planner. Dropping first makes the retry actually rebuild.

2. Autovacuum tuning on the two journal tables. The drain deletes tens of
   GB of rows whose bytes live almost entirely in TOAST; at the default
   ``autovacuum_vacuum_scale_factor = 0.2`` a 54 GB table accrues ~20% dead
   tuples before the first vacuum fires, so reclaimed space would lag the
   drain by weeks. 0.01 + no cost delay keeps vacuum roughly in step with
   the pruner. Per-table only; no global settings are touched. NOTE:
   deletion + vacuum makes space REUSABLE — returning it to the OS is a
   separate supervised pg_repack ceremony (eumemic-ops).

   The ALTERs take SHARE UPDATE EXCLUSIVE, which is online-safe but queues
   behind any running (anti-wraparound) autovacuum on the same table — and
   everything behind the ALTER then queues on it. ``lock_timeout`` +
   bounded retries make that a fail-fast loop instead of a wedged deploy.
"""

import time

from alembic import op

revision = "0172"
down_revision = "0169"
branch_labels = None
depends_on = None

_INDEX = "wf_runs_archival_backlog_idx"

_RELOPTS = "(autovacuum_vacuum_scale_factor = 0.01, autovacuum_vacuum_cost_delay = 0)"
_TABLES = ("wf_run_events", "wf_run_signals")


def _alter_with_lock_timeout(sql: str, *, attempts: int = 5) -> None:
    """Run one ALTER with a 5s lock_timeout, retrying a few times.

    A plain ALTER here can sit behind an anti-wraparound autovacuum for
    hours while blocking every query behind it in the lock queue. Failing
    fast and retrying lets concurrent traffic interleave between attempts;
    if all attempts lose the race the migration fails loudly and is safe to
    re-run.
    """
    for attempt in range(1, attempts + 1):
        try:
            op.execute("SET lock_timeout = '5s'")
            op.execute(sql)
            op.execute("SET lock_timeout = DEFAULT")
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_INDEX}")
        op.execute(
            f"CREATE INDEX CONCURRENTLY {_INDEX} "
            "ON wf_runs (updated_at, id) "
            "WHERE status IN ('completed','errored','cancelled') "
            "AND (archived_at IS NULL OR terminal_summary IS NULL)"
        )
        for table in _TABLES:
            _alter_with_lock_timeout(f"ALTER TABLE {table} SET {_RELOPTS}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_INDEX}")
        for table in _TABLES:
            _alter_with_lock_timeout(
                f"ALTER TABLE {table} RESET "
                "(autovacuum_vacuum_scale_factor, autovacuum_vacuum_cost_delay)"
            )
