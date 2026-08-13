"""Runtime inbound approval grants ledger.

Revision ID: 0164
Revises: 0158
"""

from alembic import op

revision = "0164"
down_revision = "0158"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE inbound_grants (
            id text PRIMARY KEY DEFAULT gen_random_uuid()::text,
            account_id text NOT NULL REFERENCES accounts(id),
            connection_id text NOT NULL REFERENCES connections(id) ON DELETE CASCADE,
            chat_id text NOT NULL,
            status text NOT NULL CHECK (status IN ('pending', 'active', 'revoked')),
            approved_by text REFERENCES accounts(id),
            approved_at timestamptz,
            approved_via_channel text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE UNIQUE INDEX inbound_grants_live_uniq
            ON inbound_grants (connection_id, chat_id) WHERE status <> 'revoked';
        CREATE INDEX inbound_grants_pending_gc_idx
            ON inbound_grants (created_at) WHERE status = 'pending';
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE inbound_grants")
