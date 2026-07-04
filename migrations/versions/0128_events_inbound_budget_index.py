"""Add the ``events_inbound_budget_idx`` partial index behind the
per-counterparty inbound rate budget's rolling-window count (#1557).

``_count_recent_inbounds`` (``src/aios/services/inbound_budget.py``) is the
rolling-window aggregate the inbound rate budget reads once per admitted
inbound at the connector-write boundary. Its ``WHERE`` leads with the two
equality predicates ``(account_id, orig_channel)`` — and *neither* leads any
existing index on ``events`` (every ``CREATE INDEX … ON events`` leads with
``session_id``; ``orig_channel`` was added unindexed by mig 0017). The
planner's only access path was therefore a sequential scan of the entire
``events`` log — the highest-write, fastest-growing table — paid synchronously
on the hot inbound admission path the budget exists to keep cheap.

This migration adds the composite partial index that matches the query's
access path:

* Leading ``(account_id, orig_channel)`` — the two equality predicates — gives
  the planner an equality seek into the matching rows, replacing the full scan;
  ``account_id`` first preserves per-tenant scoping as the outermost key.
* Trailing ``created_at`` lets the rolling-window range be satisfied within the
  same index, so the count is an index-range over the seeked prefix rather than
  a heap filter.
* The partial predicate mirrors the query's ``_INFERENCE_BEARING_PREDICATE``
  verbatim — an admitted inbound message (``kind='message'`` /
  ``role='user'``) OR a wake-bearing connector lifecycle (``kind='lifecycle'``
  stamped ``wake=True``). Keeping the partial predicate identical to the
  query's constant predicate is what lets the planner prove the index
  applicable while keeping the index to exactly the rows the budget counts (a
  small fraction of ``events``). This mirrors how 0099/0103 use partials to
  keep events-indexes narrow.

Only the sibling ``orig_channel``-keyed helper is fixed here;
``_count_recent_session_inbounds`` leads with ``session_id`` — the leading
column of ``events_session_seq_idx`` (mig 0080) — so it already gets an index
seek and is left untouched (#1557 Out of scope).

Purely additive — a single ``CREATE INDEX CONCURRENTLY`` built inside
``op.get_context().autocommit_block()`` so it never takes an ACCESS EXCLUSIVE
lock on the live-written ``events`` table (same pattern as 0069 / 0097 / 0099).
Safe in the post-deploy new-code/old-schema window: the index only speeds the
read; new-code reads/writes work against the old schema until it completes, and
it is invisible to the running container while building.

Revision ID: 0128
Revises: 0127
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0128"
down_revision: str = "0127"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_inbound_budget_idx "
            "ON events (account_id, orig_channel, created_at) "
            "WHERE (kind = 'message' AND data->>'role' = 'user') "
            "OR (kind = 'lifecycle' AND (data->>'wake')::boolean IS TRUE)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_inbound_budget_idx")
