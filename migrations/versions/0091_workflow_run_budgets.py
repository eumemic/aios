"""Add workflow run budget ceiling and child spend index.

Revision ID: 0091
Revises: 0090
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0091"
down_revision: str = "0090"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN budget_total_microusd bigint "
        "CHECK (budget_total_microusd IS NULL OR budget_total_microusd > 0)"
    )
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS sessions_parent_run_idx "
            "ON sessions (parent_run_id) WHERE parent_run_id IS NOT NULL"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS sessions_parent_run_idx")
    op.execute("ALTER TABLE wf_runs DROP COLUMN budget_total_microusd")
