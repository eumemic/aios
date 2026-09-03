"""Persist filesystem-reset notices across snapshot pressure reclamation.

Revision ID: 0180
Revises: 0179
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0180"
down_revision: str | None = "0179"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("snapshot_reset_pending_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "snapshot_reset_pending_ready",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "snapshot_reset_pending_ready")
    op.drop_column("sessions", "snapshot_reset_pending_reason")
