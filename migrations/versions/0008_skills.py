"""Skills system: versioned knowledge bundles for agents.

Adds two tables for skill management:

1. ``skills`` — named skill resources with versioning metadata.
2. ``skill_versions`` — immutable version snapshots containing the skill
   files (SKILL.md + optional scripts/reference docs) as JSONB.

Also adds a ``skills`` JSONB column to ``agents`` and ``agent_versions``
to store skill reference bindings (``[{skill_id, version}]``), following
the same pattern as the existing ``tools`` JSONB column.

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── skills ──────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE skills (
            id              text PRIMARY KEY,
            display_title   text NOT NULL,
            latest_version  integer NOT NULL DEFAULT 1,
            created_at      timestamptz NOT NULL DEFAULT now(),
            updated_at      timestamptz NOT NULL DEFAULT now(),
            archived_at     timestamptz
        )
    """)

    # ── skill versions ──────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE skill_versions (
            skill_id    text NOT NULL REFERENCES skills(id),
            version     integer NOT NULL,
            directory   text NOT NULL,
            name        text NOT NULL,
            description text NOT NULL DEFAULT '',
            files       jsonb NOT NULL,
            created_at  timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (skill_id, version)
        )
    """)

    # ── skills JSONB on agents ──────────────────────────────────────────
    op.execute(
        "ALTER TABLE agents ADD COLUMN skills jsonb NOT NULL DEFAULT '[]'::jsonb"
    )
    op.execute(
        "ALTER TABLE agent_versions ADD COLUMN skills jsonb NOT NULL DEFAULT '[]'::jsonb"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS skills")
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS skills")
    op.execute("DROP TABLE IF EXISTS skill_versions")
    op.execute("DROP TABLE IF EXISTS skills")
