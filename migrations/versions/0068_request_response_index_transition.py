"""Transition the request-response partial index across the event rename.

Block-2 R1 (revision 0067, committed earlier on this branch) created
``events_workflow_child_done_idx`` keyed on the ``workflow_child_done`` lifecycle
event. R4 renamed that event to ``request_response`` and reworked the index to
``events_request_response_idx`` on ``(session_id, (data->>'request_id'))``.

A fresh database gets the new index directly from 0067 (its file now creates it),
so for that DB this revision is a no-op. But a database that applied the *original*
0067 records ``alembic_version = 0067`` and would never pick up the rename — it
would keep the now-dead ``workflow_child_done`` index and miss the new one. This
revision makes the transition robust for both: drop the old index if present, and
create the new one if absent. Both statements are idempotent, so it is safe on a
fresh DB too.

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
