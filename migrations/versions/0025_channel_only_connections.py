"""Allow connections without legacy MCP projection fields.

Connections are the inbound channel-account identity. ``mcp_url`` and
``vault_id`` are optional compatibility fields for older connection-projected
MCP setups; normal connector MCP servers are declared on agents.

Revision ID: 0025
Revises: 0024
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0025"
down_revision: str = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE connections ALTER COLUMN mcp_url DROP NOT NULL")
    op.execute("ALTER TABLE connections ALTER COLUMN vault_id DROP NOT NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE connections ALTER COLUMN vault_id SET NOT NULL")
    op.execute("ALTER TABLE connections ALTER COLUMN mcp_url SET NOT NULL")
