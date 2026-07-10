"""Reverse indexes for edge-owned lifetime cascade. Revision ID: 0141."""

from collections.abc import Sequence

from alembic import op

revision: str = "0141"
down_revision: str = "0140"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS wf_runs_parent_run_idx ON wf_runs (parent_run_id) WHERE parent_run_id IS NOT NULL"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS sessions_live_parent_run_idx ON sessions (parent_run_id) WHERE archived_at IS NULL AND parent_run_id IS NOT NULL"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS sessions_live_parent_run_idx")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS wf_runs_parent_run_idx")
