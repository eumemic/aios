"""Add rescheduling to session status check constraint.

Anthropic Managed Agents support a ``rescheduling`` session status for
transient errors that will auto-retry. This migration widens the
``sessions_status_check`` constraint to allow the new value.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_status_check;"
    )
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_status_check "
        "CHECK (status IN ('running', 'idle', 'rescheduling', 'terminated'));"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_status_check;"
    )
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_status_check "
        "CHECK (status IN ('running', 'idle', 'terminated'));"
    )
