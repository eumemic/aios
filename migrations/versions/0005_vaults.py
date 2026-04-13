"""Vault and multi-credential system.

Adds three tables for MCP/service credential management:

1. ``vaults`` — named collections of credentials, workspace-scoped.
2. ``vault_credentials`` — encrypted credentials keyed by MCP server URL.
   One active credential per URL per vault (partial unique index).
3. ``session_vaults`` — junction table binding sessions to vaults with
   rank-based ordering for first-match credential resolution.

Also drops ``credential_id`` from ``agents`` and ``agent_versions`` — model
API auth is now handled by LiteLLM's standard env var resolution.

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── vaults ───────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE vaults (
            id           text PRIMARY KEY,
            display_name text NOT NULL,
            metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at   timestamptz NOT NULL DEFAULT now(),
            updated_at   timestamptz NOT NULL DEFAULT now(),
            archived_at  timestamptz
        )
    """)

    # ── vault credentials ────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE vault_credentials (
            id              text PRIMARY KEY,
            vault_id        text NOT NULL REFERENCES vaults(id),
            display_name    text,
            mcp_server_url  text NOT NULL,
            auth_type       text NOT NULL
                CHECK (auth_type IN ('mcp_oauth', 'static_bearer')),
            ciphertext      bytea NOT NULL,
            nonce           bytea NOT NULL,
            metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at      timestamptz NOT NULL DEFAULT now(),
            updated_at      timestamptz NOT NULL DEFAULT now(),
            archived_at     timestamptz
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX vault_credentials_url_uniq
            ON vault_credentials (vault_id, mcp_server_url)
            WHERE archived_at IS NULL
    """)
    op.execute("""
        CREATE INDEX vault_credentials_vault_idx
            ON vault_credentials (vault_id)
    """)

    # ── session-vault junction ───────────────────────────────────────────
    op.execute("""
        CREATE TABLE session_vaults (
            session_id text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            vault_id   text NOT NULL REFERENCES vaults(id),
            rank       integer NOT NULL,
            PRIMARY KEY (session_id, vault_id)
        )
    """)

    # ── drop credential_id from agents ───────────────────────────────────
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS credential_id")
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS credential_id")


def downgrade() -> None:
    op.execute("ALTER TABLE agents ADD COLUMN credential_id text REFERENCES credentials(id)")
    op.execute(
        "ALTER TABLE agent_versions ADD COLUMN credential_id text REFERENCES credentials(id)"
    )
    op.execute("DROP TABLE IF EXISTS session_vaults")
    op.execute("DROP TABLE IF EXISTS vault_credentials")
    op.execute("DROP TABLE IF EXISTS vaults")
