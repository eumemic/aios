"""Partial index on the ``workflow_child_done`` completion marker.

A workflow agent child signals logical completion by appending a single
``workflow_child_done`` lifecycle event (see ``tools/workflow_completion.py``).
Three readers resolve that exact predicate and all benefit:

* ``sweep.COMPLETED_CHILD_SESSIONS_SQL`` (``find_sessions_needing_inference``)
  scans marker rows **cross-session** every sweep pass to subtract soft-terminal
  children from the wake set — previously unindexed for this predicate.
* the in-worker workflow-child reaper's candidate query (Block 2.F) selects
  marker-present children to archive.
* ``queries.read_workflow_child_done`` reads one child's marker on every parent
  run-step harvest.

Markers are rare relative to total events (one per finished workflow child), so
this is a small partial index. Built with ``CREATE INDEX CONCURRENTLY`` (outside
a transaction via ``autocommit_block``) so it never takes an ACCESS EXCLUSIVE
lock on the live-written ``events`` table — same pattern as migrations 0023 /
0062 / 0065.

Revision ID: 0067
Revises: 0066
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0067"
down_revision: str = "0066"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_workflow_child_done_idx "
            "ON events (session_id) "
            "WHERE kind = 'lifecycle' AND data->>'event' = 'workflow_child_done'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_workflow_child_done_idx")
