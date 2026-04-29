"""Drop MCP config fields from connections.

Connections now carry only inbound channel-account identity. MCP server
configuration and credentials live on agents and session vaults.

Revision ID: 0026
Revises: 0025
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0026"
down_revision: str = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE connections DROP COLUMN vault_id")
    op.execute("ALTER TABLE connections DROP COLUMN mcp_url")


def downgrade() -> None:
    op.execute("ALTER TABLE connections ADD COLUMN mcp_url text")
    op.execute("ALTER TABLE connections ADD COLUMN vault_id text REFERENCES vaults(id)")
