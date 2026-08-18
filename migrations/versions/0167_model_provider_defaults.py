"""Add generalized model-provider defaults and optimistic versioning.

Revision ID: 0167
Revises: 0166
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0167"
down_revision: str = "0166"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE model_providers "
        "ADD COLUMN litellm_defaults jsonb NOT NULL DEFAULT '{}'::jsonb, "
        "ADD COLUMN version integer NOT NULL DEFAULT 1"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE model_providers DROP COLUMN version, DROP COLUMN litellm_defaults")
