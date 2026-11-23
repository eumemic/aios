"""Canonical model routes owned by model-provider configuration.

Revision ID: 0196
Revises: 0195
"""

from __future__ import annotations

from alembic import op

revision = "0196"
down_revision = "0195"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE model_providers "
        "ADD COLUMN model_routes jsonb NOT NULL DEFAULT '{}'::jsonb"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE model_providers DROP COLUMN model_routes")
