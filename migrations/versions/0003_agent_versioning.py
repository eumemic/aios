"""Agent versioning and session mutability.

Adds version tracking to agents and allows sessions to pin to a specific
agent version or float on "latest" (the default).

- agents: add ``version`` column (integer, default 1)
- agent_versions: new table storing a full config snapshot per version
- sessions: add ``agent_version`` column (nullable integer; NULL = latest)

Existing agents become version 1. Existing sessions get NULL agent_version
(they were already implicitly using latest).

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add version column to agents (existing rows become version 1).
    op.execute("ALTER TABLE agents ADD COLUMN version integer NOT NULL DEFAULT 1;")

    # 2. Create agent_versions table.
    op.execute(
        """
        CREATE TABLE agent_versions (
            agent_id        text NOT NULL REFERENCES agents(id),
            version         integer NOT NULL,
            model           text NOT NULL,
            system          text NOT NULL DEFAULT '',
            tools           jsonb NOT NULL DEFAULT '[]'::jsonb,
            credential_id   text REFERENCES credentials(id),
            window_min      integer NOT NULL DEFAULT 50000,
            window_max      integer NOT NULL DEFAULT 150000,
            created_at      timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (agent_id, version)
        );
        """
    )

    # 3. Backfill: snapshot every existing agent as version 1.
    op.execute(
        """
        INSERT INTO agent_versions (agent_id, version, model, system, tools,
                                    credential_id, window_min, window_max, created_at)
        SELECT id, 1, model, system, tools,
               credential_id, window_min, window_max, created_at
          FROM agents;
        """
    )

    # 4. Add agent_version to sessions (NULL = latest).
    op.execute("ALTER TABLE sessions ADD COLUMN agent_version integer;")


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS agent_version;")
    op.execute("DROP TABLE IF EXISTS agent_versions;")
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS version;")
