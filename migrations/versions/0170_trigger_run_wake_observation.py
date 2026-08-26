"""Record whether a trigger fire woke its owner.

Revision ID: 0170
Revises: 0171
"""

from alembic import op

revision = "0170"
down_revision = "0173"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE trigger_runs ADD COLUMN woke_owner boolean NOT NULL DEFAULT false")
    op.execute(
        "CREATE INDEX trigger_runs_recent_wakes "
        "ON trigger_runs (trigger_id, finished_at DESC) "
        "WHERE trigger_context = 'cron'"
    )
    # A workflow child can wake before the trigger runner has received the
    # launched run id and written its audit row. Persist the effect at the run
    # boundary so the later audit insert can reconcile that ordering race.
    op.execute(
        "CREATE TABLE workflow_run_owner_wakes ("
        "workflow_run_id text PRIMARY KEY REFERENCES wf_runs(id) ON DELETE CASCADE, "
        "observed_at timestamptz NOT NULL DEFAULT now()"
        ")"
    )


def downgrade() -> None:
    op.execute("DROP TABLE workflow_run_owner_wakes")
    op.execute("DROP INDEX trigger_runs_recent_wakes")
    op.execute("ALTER TABLE trigger_runs DROP COLUMN woke_owner")
