"""Add the production-safe terminal-detail eligibility index.

Revision ID: 0160
Revises: 0159
"""

from alembic import op

revision = "0160"
down_revision = "0159"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS wf_runs_prune_eligibility_idx "
            "ON wf_runs (archived_at, id) WHERE archived_at IS NOT NULL "
            "AND journal_pruned_at IS NULL"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS wf_runs_prune_eligibility_idx")
