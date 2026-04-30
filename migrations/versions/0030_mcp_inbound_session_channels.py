"""Add MCP inbound session channel state.

Revision ID: 0030
Revises: 0029
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0030"
down_revision: str = "0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE session_channels (
            id                text PRIMARY KEY,
            session_id        text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            mcp_server_name   text NOT NULL,
            mcp_server_url    text NOT NULL,
            account_id        text NOT NULL,
            path              text NOT NULL,
            address           text NOT NULL,
            display_name      text,
            notification_mode text NOT NULL DEFAULT 'focal_candidate'
                CHECK (notification_mode IN ('focal_candidate', 'silent')),
            metadata          jsonb NOT NULL DEFAULT '{}'::jsonb,
            last_seen_at      timestamptz NOT NULL DEFAULT now(),
            created_at        timestamptz NOT NULL DEFAULT now(),
            updated_at        timestamptz NOT NULL DEFAULT now(),
            archived_at       timestamptz
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX session_channels_identity_uniq
            ON session_channels (session_id, mcp_server_name, account_id, path)
            WHERE archived_at IS NULL
    """)
    op.execute("""
        CREATE UNIQUE INDEX session_channels_address_uniq
            ON session_channels (session_id, address)
            WHERE archived_at IS NULL
    """)
    op.execute("""
        CREATE INDEX session_channels_session_idx
            ON session_channels (session_id)
            WHERE archived_at IS NULL
    """)

    op.execute("""
        CREATE TABLE inbound_mcp_cursors (
            session_id          text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            mcp_server_name     text NOT NULL,
            mcp_server_url      text NOT NULL,
            vault_credential_id text NOT NULL REFERENCES vault_credentials(id) ON DELETE CASCADE,
            account_id          text NOT NULL,
            last_event_id       text,
            updated_at          timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, mcp_server_name, vault_credential_id, account_id)
        )
    """)

    op.execute("""
        CREATE TABLE inbound_mcp_receipts (
            session_id      text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            mcp_server_name text NOT NULL,
            account_id      text NOT NULL,
            event_id        text NOT NULL,
            event_row_id    text REFERENCES events(id) ON DELETE SET NULL,
            created_at      timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, mcp_server_name, account_id, event_id)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS inbound_mcp_receipts")
    op.execute("DROP TABLE IF EXISTS inbound_mcp_cursors")
    op.execute("DROP TABLE IF EXISTS session_channels")
