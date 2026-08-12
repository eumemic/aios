"""Add image-aware token baseline v2 state.

Revision ID: 0159
Revises: 0158
"""

from alembic import op

revision = "0159"
down_revision = "0158"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE events "
        "ADD COLUMN cumulative_image_mass bigint NOT NULL DEFAULT 0, "
        "ADD COLUMN token_baseline_v smallint NOT NULL DEFAULT 1"
    )
    op.execute("ALTER TABLE sessions ADD COLUMN token_baseline_v smallint NOT NULL DEFAULT 1")


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN token_baseline_v")
    op.execute("ALTER TABLE events DROP COLUMN token_baseline_v, DROP COLUMN cumulative_image_mass")
