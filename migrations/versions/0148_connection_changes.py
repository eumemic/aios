"""Add durable sequenced connection discovery ledger.

Revision ID: 0148
Revises: 0147
"""

from alembic import op

revision = "0148"
down_revision = "0147"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE connection_changes (
            seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES accounts(id),
            connector TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('added', 'removed')),
            connection_id TEXT NOT NULL,
            external_account_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX connection_changes_discovery_idx
        ON connection_changes (account_id, connector, seq)
    """)


def downgrade() -> None:
    op.drop_table("connection_changes")
