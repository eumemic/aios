"""Runtime inbound approval grants ledger.

Revision ID: 0164
Revises: 0165
"""

from alembic import op

revision = "0164"
# Re-parented 0159 -> 0165 (fix round, 2026-08-17). This branch was authored when
# 0159 was master's head; master has since advanced 0159 -> 0161 -> 0162 -> 0163 ->
# 0165. A git rebase moves the FILE but does NOT re-parent a migration: after the
# rebase both 0164 (parent 0159) and 0165 (parent 0163) claimed to be heads, so the
# ladder forked and `alembic upgrade head` fails with "Multiple head revisions are
# present". Same trap as 0161's re-parent note below-history, and aios#2172/#2126.
# The revision NUMBER stays 0164 (it is already stamped in the PR's own history and
# ids are opaque to alembic); only the parent pointer moves so the chain stays linear.
# Verified before editing: master has exactly ONE head, 0165.
down_revision = "0165"
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
