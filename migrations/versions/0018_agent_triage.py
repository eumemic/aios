"""Agent triage gate: optional cheap-model gate before main inference.

Adds a nullable ``triage`` jsonb column to both ``agents`` (current) and
``agent_versions`` (history). A NULL value means "no triage — always
respond," preserving prior behavior.

The JSON shape is ``{"model": str, "system": str}``; validation happens
in the Pydantic layer, not in the DB.

Revision ID: 0018
Revises: 0017
Create Date: 2026-04-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0018"
down_revision: str = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE agents ADD COLUMN triage jsonb;")
    op.execute("ALTER TABLE agent_versions ADD COLUMN triage jsonb;")


def downgrade() -> None:
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS triage;")
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS triage;")
