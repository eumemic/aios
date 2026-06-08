"""Partial index on the workflow child's completion event (superseded by 0069).

Historical: when this revision shipped, a workflow child answered its request with
a ``workflow_child_done`` lifecycle event, and this index made the harvest's read a
point lookup. R4 later renamed that event to ``request_response``; **migration 0069
drops this index and creates ``events_request_response_idx`` in its place**, so on a
fresh database the index built here lives only until 0069 runs. This body is left
as-shipped (it still builds the old index) so 0069 can transition every database
uniformly — see 0069 for the rationale.

Built with ``CREATE INDEX CONCURRENTLY`` (outside a transaction via
``autocommit_block``) so it never takes an ACCESS EXCLUSIVE lock on the
live-written ``events`` table — same pattern as migrations 0023 / 0062 / 0065.

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
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_workflow_child_done_idx "
            "ON events (session_id) "
            "WHERE kind = 'lifecycle' AND data->>'event' = 'workflow_child_done'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_workflow_child_done_idx")
