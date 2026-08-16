"""Add caller-selected terminal archival to workflow runs.

Revision ID: 0190
Revises: 0159
"""
from alembic import op

revision = "0190"
down_revision = "0159"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN auto_archive_on_completion "
        "boolean NOT NULL DEFAULT false"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS auto_archive_on_completion")
