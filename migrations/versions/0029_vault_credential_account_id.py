"""Add optional account identity to vault credentials.

Revision ID: 0029
Revises: 0028
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0029"
down_revision: str = "0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("vault_credentials", sa.Column("account_id", sa.Text(), nullable=True))
    op.create_check_constraint(
        "vault_credentials_account_id_segment_ck",
        "vault_credentials",
        "account_id IS NULL OR (account_id <> '' AND account_id NOT LIKE '%/%')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "vault_credentials_account_id_segment_ck",
        "vault_credentials",
        type_="check",
    )
    op.drop_column("vault_credentials", "account_id")
