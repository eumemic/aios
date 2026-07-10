"""Add soft creator provenance to tenant resource families.

Revision ID: 0146
Revises: 0145
"""
from alembic import op

revision = "0146"
down_revision = "0145"
branch_labels = None
depends_on = None

_TABLES = (
    "agents", "workflows", "skills", "sessions", "vaults",
    "vault_credentials", "environments", "connections", "triggers",
)


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN created_by_type text")
        op.execute(f"ALTER TABLE {table} ADD COLUMN created_by_ref text")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {table}_created_by_ck CHECK ("
            "(created_by_type IS NULL AND created_by_ref IS NULL) OR "
            "(created_by_type IN ('session_actor', 'api_actor') AND created_by_ref IS NOT NULL))"
        )


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_created_by_ck")
        op.execute(f"ALTER TABLE {table} DROP COLUMN created_by_ref")
        op.execute(f"ALTER TABLE {table} DROP COLUMN created_by_type")
