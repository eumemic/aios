"""Partial indexes for the model-dispatch park crash-recovery sweep (#1707).

``find_unharvested_model_dispatch_parks`` (#1635) runs on every 30s periodic
sweep, cross-session. Its outer predicate —

    kind='span' AND data->>'event'='model_workflow_park'

— had no supporting index, so on the periodic path (``session_id is None``, empty
``scope_clause``) nothing indexed restricted the scan and it fell to a sequential
scan of ``events`` (the largest, fastest-growing table). The two ``NOT EXISTS``
anti-joins probe ``model_workflow_harvest_end`` / ``model_workflow_harvest`` spans,
which were likewise unindexed.

The repo convention is a dedicated partial index per sweep/lifecycle span
predicate (``request_opened`` → 0099, ``workflow_child_done`` → 0068/0069,
``tool_confirmed`` → 0065, ``model_request_end`` → 0024). #1635 added the park
scan without its companion index — a straightforward omission this migration
closes with three partial indexes:

* ``events_model_workflow_park_idx`` — the required deliverable. Keyed
  ``(session_id, seq DESC)`` so ``DISTINCT ON (session_id) … ORDER BY session_id,
  seq DESC`` reads the latest park per session straight off the index and the
  per-session scoped path can seek on ``session_id``.
* ``events_model_workflow_harvest_end_idx`` /
  ``events_model_workflow_harvest_idx`` — companions for the anti-join probes,
  keyed ``(session_id, account_id, (data->>'run_id'))`` to match their
  ``session_id = … AND account_id = … AND data->>'run_id' = …`` correlation and
  serve them as index scans rather than seq scans.

Built with ``CREATE INDEX CONCURRENTLY`` (0024's pattern) to avoid an ACCESS
EXCLUSIVE lock on the live ``events`` table; alembic's implicit BEGIN is disabled
via ``autocommit_block()`` so CONCURRENTLY is allowed. ``IF NOT EXISTS`` /
``IF EXISTS`` make each statement idempotent — a CONCURRENTLY build that fails
partway leaves an INVALID index behind, and the guard lets a re-run proceed.

Revision ID: 0131
Revises: 0130
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0131"
down_revision: str = "0130"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


PARK_INDEX = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_model_workflow_park_idx "
    "ON events (session_id, seq DESC) "
    "WHERE kind = 'span' AND data->>'event' = 'model_workflow_park'"
)

HARVEST_END_INDEX = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_model_workflow_harvest_end_idx "
    "ON events (session_id, account_id, (data->>'run_id')) "
    "WHERE kind = 'span' AND data->>'event' = 'model_workflow_harvest_end'"
)

HARVEST_INDEX = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_model_workflow_harvest_idx "
    "ON events (session_id, account_id, (data->>'run_id')) "
    "WHERE kind = 'span' AND data->>'event' = 'model_workflow_harvest'"
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(PARK_INDEX)
        op.execute(HARVEST_END_INDEX)
        op.execute(HARVEST_INDEX)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_model_workflow_harvest_idx")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_model_workflow_harvest_end_idx")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_model_workflow_park_idx")
