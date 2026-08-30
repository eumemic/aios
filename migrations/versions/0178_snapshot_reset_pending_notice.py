"""Persist filesystem-reset notices across snapshot pressure reclamation.

Revision ID: 0178
Revises: 0177
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0178"
down_revision: str | None = "0177"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("snapshot_reset_pending_reason", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sessions", "snapshot_reset_pending_reason")
