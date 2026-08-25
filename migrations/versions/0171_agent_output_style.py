"""Add ``output_style`` column to agents + agent_versions.

Purely additive, default 'default' -- every existing agent keeps current
behavior; versioned like every other config field. No CHECK constraint --
the ``OutputStyle`` Literal on the pydantic models is the single
validation point (0139/0111 precedent). Mechanism prose lives in
``aios.harness.concise``.

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
    op.execute("ALTER TABLE agents ADD COLUMN output_style text NOT NULL DEFAULT 'default';")
    op.execute(
        "ALTER TABLE agent_versions ADD COLUMN output_style text NOT NULL DEFAULT 'default';"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS output_style;")
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS output_style;")
