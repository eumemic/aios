"""Add ``concise`` boolean column to agents + agent_versions.

Purely additive, default false -- every existing agent keeps current
behavior; versioned like every other config field. Mechanism prose
lives in ``aios.harness.concise``.

Revision ID: 0171
Revises: 0169
"""

from __future__ import annotations

from alembic import op

revision = "0171"
down_revision = "0169"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE agents ADD COLUMN concise boolean NOT NULL DEFAULT false;")
    op.execute("ALTER TABLE agent_versions ADD COLUMN concise boolean NOT NULL DEFAULT false;")


def downgrade() -> None:
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS concise;")
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS concise;")
