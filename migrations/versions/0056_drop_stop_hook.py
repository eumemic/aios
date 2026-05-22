"""Drop the orphaned ``stop_hook`` column left by the #603 stop-hook revert.

#603 added ``sessions.stop_hook`` (migration 0055). #613/#615 reverted the
stop-hook feature in code but the production DB kept the column and the 0055
stamp. This migration brings the schema back in line with the reverted code.

Revision ID: 0056
Revises: 0055
Create Date: 2026-05-22
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0056"
down_revision: str = "0055"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS stop_hook")


def downgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN stop_hook jsonb")
