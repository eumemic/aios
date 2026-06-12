"""Add per-account spend meter.

Revision ID: 0090
Revises: 0089
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0090"
down_revision: str = "0089"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE accounts ADD COLUMN spent_microusd bigint NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE sessions ADD COLUMN cost_microusd bigint NOT NULL DEFAULT 0")


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN cost_microusd")
    op.execute("ALTER TABLE accounts DROP COLUMN spent_microusd")
