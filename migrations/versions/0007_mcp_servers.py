"""MCP server declarations on agents.

Adds an ``mcp_servers`` JSONB column to both ``agents`` and
``agent_versions`` for declaring remote MCP servers that provide
additional tools via the Model Context Protocol. Default is an
empty array so existing rows are backward-compatible.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agents ADD COLUMN mcp_servers jsonb NOT NULL DEFAULT '[]'::jsonb;"
    )
    op.execute(
        "ALTER TABLE agent_versions ADD COLUMN mcp_servers jsonb NOT NULL DEFAULT '[]'::jsonb;"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS mcp_servers;")
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS mcp_servers;")
