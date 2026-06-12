"""Soft-archive workflows and release archived names.

Workflow definitions follow the never-delete lifecycle: archived rows stay
addressable by id, disappear from lists, and no longer claim their tenant/name.

Revision ID: 0089
Revises: 0088
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0089"
down_revision: str = "0088"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE workflows ADD COLUMN archived_at timestamptz")
    op.execute("ALTER TABLE workflows DROP CONSTRAINT workflows_account_id_name_key")
    op.execute(
        "CREATE UNIQUE INDEX workflows_account_id_name_key "
        "ON workflows (account_id, name) WHERE archived_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX workflows_account_id_name_key")
    op.execute(
        "ALTER TABLE workflows ADD CONSTRAINT workflows_account_id_name_key "
        "UNIQUE (account_id, name)"
    )
    op.execute("ALTER TABLE workflows DROP COLUMN archived_at")
