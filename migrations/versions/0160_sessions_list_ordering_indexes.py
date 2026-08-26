"""Supporting indexes for keyset-paginated ``list_sessions`` ordering.

PR #1939 added ``order_by`` to ``list_sessions`` and rewrote the query to
order by ``(<order column> DESC, id DESC)`` instead of the old ``id DESC``.
That rewrite silently regressed the DEFAULT path — ``order_by="created_at"``,
which every existing caller uses and nobody opted into. Measured against
PostgreSQL 15 with a 60,000-session tenant:

    OLD  Index Scan Backward using sessions_pkey, 51 rows   →  ~0.4 ms
    NEW  Seq Scan (rows=60000) + top-N heapsort            →  ~24  ms

The regression is NOT caused by the CTE wrapper (Postgres inlines the
non-materialised ``sessions_page`` CTE, so ``created_at`` ordering can still
ride an index). It is caused by there being **no index whose sort order
matches ``(account_id, created_at DESC, id DESC)``** filtered to live rows.
With the matching index the planner drops back to an Index Scan and the fast
exit returns — ~0.9 ms even with the CTE present, measured on the same tenant.

This migration adds that index (and the ``updated_at`` analog). It is
partial (``archived_at IS NULL``) to match the default listing's WHERE clause
and to stay small; the archive-inclusive listings (``parent_run_id`` /
``status="archived"``) are narrow by their own predicates and don't need it.

There is NO index for the ``order_by="last_event_at"`` path: ``last_event_at``
is a correlated subquery over the event log, not a stored column, so no index
on ``sessions`` can order by it. That path is inherently O(tenant) per page and
this migration does not pretend otherwise — materialising ``last_event_at`` as
a maintained column is the only real fix and is deliberately out of scope here.

``CREATE INDEX CONCURRENTLY`` (no table lock on a hot table); the migration
runs outside a transaction, matching the project's other CONCURRENTLY
migrations (e.g. 0091, 0141).

Revision ID: 0160
Revises: 0159
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0160"
down_revision: str = "0159"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS sessions_account_created_id_idx "
            "ON sessions (account_id, created_at DESC, id DESC) WHERE archived_at IS NULL"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS sessions_account_updated_id_idx "
            "ON sessions (account_id, updated_at DESC, id DESC) WHERE archived_at IS NULL"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS sessions_account_created_id_idx")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS sessions_account_updated_id_idx")
