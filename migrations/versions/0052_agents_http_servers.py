"""Add ``http_servers`` JSONB column to ``agents`` + ``agent_versions``.

Stores the agent-declared list of authenticated HTTP endpoints
(``HttpServerSpec``: name + base_url + route allowlist + per-route
permission policy) the agent can reach via the ``http_request``
built-in tool.  Defaults to ``'[]'::jsonb`` so existing rows Just Work;
changing the list creates a new agent version (same as ``mcp_servers``).

Part of #465 (`http_servers` agent-config primitive).

Revision ID: 0052
Revises: 0051
Create Date: 2026-05-15
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0052"
down_revision: str = "0051"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE agents ADD COLUMN http_servers jsonb NOT NULL DEFAULT '[]'::jsonb;")
    op.execute(
        "ALTER TABLE agent_versions ADD COLUMN http_servers jsonb NOT NULL DEFAULT '[]'::jsonb;"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS http_servers;")
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS http_servers;")
