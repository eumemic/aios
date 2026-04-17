"""Routing infrastructure: connections, channel bindings, routing rules.

Phase 1 of the connectors/channels design (issue #30).

* ``connections`` — registered connector+account instances. Each one
  identifies an external messaging account and the MCP URL connectors
  expose to the worker for sending replies.
* ``channel_bindings`` — explicit ``address → session_id`` map.  When an
  inbound message arrives, the resolver checks here first.
* ``routing_rules`` — fallback when no binding exists.  A rule's
  ``prefix`` is matched against the address segment-aware, longest-prefix
  wins.  ``target`` is one of ``agent:<id>[@<version>]`` or
  ``session:<id>``; for ``agent:`` targets, ``session_params`` carries
  the args used to spin up a fresh session at resolve time.

Revision ID: 0015
Revises: 0014
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0015"
down_revision: str = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── connections ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE connections (
            id          text PRIMARY KEY,
            connector   text NOT NULL,
            account     text NOT NULL,
            mcp_url     text NOT NULL,
            vault_id    text NOT NULL REFERENCES vaults(id),
            metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at  timestamptz NOT NULL DEFAULT now(),
            updated_at  timestamptz NOT NULL DEFAULT now(),
            archived_at timestamptz
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX connections_connector_account_uniq
            ON connections (connector, account) WHERE archived_at IS NULL
    """)

    # ── channel bindings ─────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE channel_bindings (
            id          text PRIMARY KEY,
            address     text NOT NULL,
            session_id  text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            created_at  timestamptz NOT NULL DEFAULT now(),
            updated_at  timestamptz NOT NULL DEFAULT now(),
            archived_at timestamptz
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX channel_bindings_address_uniq
            ON channel_bindings (address) WHERE archived_at IS NULL
    """)
    op.execute("""
        CREATE INDEX channel_bindings_session_id
            ON channel_bindings (session_id) WHERE archived_at IS NULL
    """)

    # ── routing rules ────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE routing_rules (
            id             text PRIMARY KEY,
            prefix         text NOT NULL,
            target         text NOT NULL,
            session_params jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at     timestamptz NOT NULL DEFAULT now(),
            updated_at     timestamptz NOT NULL DEFAULT now(),
            archived_at    timestamptz
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX routing_rules_prefix_uniq
            ON routing_rules (prefix) WHERE archived_at IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS routing_rules")
    op.execute("DROP TABLE IF EXISTS channel_bindings")
    op.execute("DROP TABLE IF EXISTS connections")
