"""Record whether a trigger fire woke its owner.

Revision ID: 0167
Revises: 0166
"""

from alembic import op

revision = "0167"
down_revision = "0166"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE trigger_runs ADD COLUMN woke_owner boolean NOT NULL DEFAULT false"
    )
    op.execute(
        "CREATE INDEX trigger_runs_recent_wakes "
        "ON trigger_runs (trigger_id, finished_at DESC) "
        "WHERE trigger_context = 'cron'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX trigger_runs_recent_wakes")
    op.execute("ALTER TABLE trigger_runs DROP COLUMN woke_owner")
