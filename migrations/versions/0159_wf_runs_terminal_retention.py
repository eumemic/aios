"""Preserve terminal workflow summaries while allowing detail expiry.

Revision ID: 0159
Revises: 0158
"""

from alembic import op

revision = "0159"
down_revision = "0158"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_runs ADD COLUMN terminal_summary jsonb")
    op.execute("ALTER TABLE wf_runs ADD COLUMN journal_pruned_at timestamptz")


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS journal_pruned_at")
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS terminal_summary")
