"""Preserve terminal workflow summaries while allowing detail expiry.

Revision ID: 0162
Revises: 0159
"""

from alembic import op

revision = "0162"
# 0161 remains on unmerged PR #2073, so it is not a valid parent on current master.
# Keep this branch's migration graph executable from the actual landed head. If #2073
# lands first, the second PR to merge must linearize its migration during its required
# rebase; pointing at an absent future revision makes every DB-backed check fail today.
down_revision = "0159"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_runs ADD COLUMN terminal_summary jsonb")
    op.execute("ALTER TABLE wf_runs ADD COLUMN journal_pruned_at timestamptz")


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS journal_pruned_at")
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS terminal_summary")
