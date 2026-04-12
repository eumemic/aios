"""Drop lease columns from sessions table.

Phase 5 replaces the custom DB-row lease protocol with procrastinate's
built-in ``lock`` parameter for mutual exclusion and heartbeat-based
crash recovery. The ``lease_worker_id`` and ``lease_expires_at`` columns
are no longer read or written by any code path.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS sessions_lease_idx;")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS lease_worker_id;")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS lease_expires_at;")


def downgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS lease_worker_id text;")
    op.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS sessions_lease_idx
            ON sessions (lease_expires_at)
         WHERE lease_worker_id IS NOT NULL;
        """
    )
