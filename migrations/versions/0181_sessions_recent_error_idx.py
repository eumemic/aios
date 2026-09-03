"""Index recent errored-session health queries.

Revision ID: 0181
Revises: 0180
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0181"
down_revision: str | None = "0180"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        # An interrupted concurrent build can leave an invalid same-named index.
        # Remove any remnant so retries always perform a usable rebuild.
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS sessions_recent_error_idx")
        op.execute(
            "CREATE INDEX CONCURRENTLY sessions_recent_error_idx "
            "ON sessions (account_id, updated_at DESC) "
            "WHERE archived_at IS NULL AND stop_reason->>'type' = 'error'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS sessions_recent_error_idx")
