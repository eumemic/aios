"""Transition the request-response partial index across the event rename.

Block-2 R1 (revision 0067) created ``events_workflow_child_done_idx`` keyed on the
``workflow_child_done`` lifecycle event. R4 renamed that event to
``request_response`` and reworked the index to ``events_request_response_idx`` on
``(session_id, (data->>'request_id'))``. 0067's own body was left untouched (it
still creates the old index), so this revision does the whole transition for EVERY
database — fresh or already stamped at 0067: drop the old index if present, create
the new one if absent. On a fresh DB the chain runs 0067 (old index) → 0068 (drop
it, create new); on a DB that applied the original 0067, only 0068 runs and flips
it. Both statements are idempotent, so re-running is safe.

``CREATE``/``DROP INDEX CONCURRENTLY`` run outside a transaction via
``autocommit_block`` so they never take an ACCESS EXCLUSIVE lock on the
live-written ``events`` table (same pattern as 0023 / 0062 / 0065 / 0067).

Revision ID: 0068
Revises: 0067
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0068"
down_revision: str = "0067"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_workflow_child_done_idx")
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_request_response_idx "
            "ON events (session_id, (data->>'request_id')) "
            "WHERE kind = 'lifecycle' AND data->>'event' = 'request_response'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_request_response_idx")
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_workflow_child_done_idx "
            "ON events (session_id) "
            "WHERE kind = 'lifecycle' AND data->>'event' = 'workflow_child_done'"
        )
