"""Index workflow-run collection recency ordering.

Revision ID: 0176
Revises: 0175

The indexes are built and removed concurrently so this migration remains online
on a populated ``wf_runs`` table. Each build drops any same-named index first:
a cancelled ``CREATE INDEX CONCURRENTLY`` can leave an invalid index behind,
and ``IF NOT EXISTS`` alone would incorrectly preserve that unusable index on
retry. Both operations run in an autocommit block because PostgreSQL forbids
concurrent index DDL inside a transaction block.
"""

from alembic import op

revision = "0176"
down_revision = "0175"
branch_labels = None
depends_on = None

_INDEXES = (
    (
        "wf_runs_account_recency_idx",
        "ON wf_runs (account_id, created_at DESC, id DESC) WHERE archived_at IS NULL",
    ),
    (
        "wf_runs_account_workflow_recency_idx",
        "ON wf_runs (account_id, workflow_id, created_at DESC, id DESC) "
        "WHERE archived_at IS NULL",
    ),
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        for name, definition in _INDEXES:
            # Remove an invalid remnant from an interrupted concurrent build so
            # rerunning the migration performs a real rebuild.
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
            op.execute(f"CREATE INDEX CONCURRENTLY {name} {definition}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name, _definition in reversed(_INDEXES):
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
