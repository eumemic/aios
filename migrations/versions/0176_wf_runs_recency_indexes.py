"""Index workflow-run collection recency ordering.

Revision ID: 0176
Revises: 0175
"""

from alembic import op

revision = "0176"
down_revision = "0175"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX wf_runs_account_recency_idx "
        "ON wf_runs (account_id, created_at DESC, id DESC) "
        "WHERE archived_at IS NULL"
    )
    op.execute(
        "CREATE INDEX wf_runs_account_workflow_recency_idx "
        "ON wf_runs (account_id, workflow_id, created_at DESC, id DESC) "
        "WHERE archived_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX wf_runs_account_workflow_recency_idx")
    op.execute("DROP INDEX wf_runs_account_recency_idx")
