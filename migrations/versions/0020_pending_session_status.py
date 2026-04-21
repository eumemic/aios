"""Add ``pending`` to the session status check constraint.

Allows ``POST /v1/sessions/:id/messages`` to flip the session from
``idle`` to ``pending`` before the worker picks up the deferred wake,
so external orchestrators can distinguish "queued but not started"
from "turn finished." See issue #39.

Revision ID: 0020
Revises: 0019
Create Date: 2026-04-21
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0020"
down_revision: str = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_status_check;")
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_status_check "
        "CHECK (status IN ('pending', 'running', 'idle', 'rescheduling', 'terminated'));"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE sessions SET status = 'idle' WHERE status = 'pending';"
    )
    op.execute("ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_status_check;")
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_status_check "
        "CHECK (status IN ('running', 'idle', 'rescheduling', 'terminated'));"
    )
