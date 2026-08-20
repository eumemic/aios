"""Canonical model routes owned by model-provider configuration.

Revision ID: 0167
Revises: 0166
"""

from __future__ import annotations

from alembic import op

revision = "0167"
down_revision = "0166"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE model_providers "
        "ADD COLUMN model_routes jsonb NOT NULL DEFAULT '{}'::jsonb"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE model_providers DROP COLUMN model_routes")
