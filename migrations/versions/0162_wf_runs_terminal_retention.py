"""Preserve terminal workflow summaries while allowing detail expiry.

Revision ID: 0162
Revises: 0158
"""

from alembic import op

revision = "0162"
# Re-parented 0158 -> 0161 (seat, 2026-08-13). Same branched-history defect as #2073's
# 0161: authored when 0158 was the tip, and 0159_connection_changes landed on master
# since -- so 0162 and 0159 both claimed 0158 as parent, giving alembic TWO HEADS, which
# fails every DB-touching check (unit included; that is what distinguishes this from the
# known-flaky e2e shard in #1967).
# Parented onto 0161 (not 0159) because this PR's 0162/0163 chain BEHIND #2073's 0161.
# A git rebase moves the file but does NOT re-parent a migration.
down_revision = "0161"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_runs ADD COLUMN terminal_summary jsonb")
    op.execute("ALTER TABLE wf_runs ADD COLUMN journal_pruned_at timestamptz")


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS journal_pruned_at")
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS terminal_summary")
