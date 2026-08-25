"""Add ``concise`` boolean column to agents + agent_versions.

Per-agent opt-in that steers the model toward short, direct output: the
harness joins a cache-stable concise-style rules block into the system
prompt and appends a one-line tail reminder to the composed payload each
step (assembly-time only — never persisted to ``agent_events``).  Default
``false`` so every existing agent keeps current behavior; versioned like
every other config field (changing it creates a new agent version).
Purely additive, safe in the new-code/old-schema deploy window (0139
precedent).

Revision ID: 0170
Revises: 0169
Create Date: 2026-08-25
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0170"
down_revision: str = "0169"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE agents ADD COLUMN concise boolean NOT NULL DEFAULT false;")
    op.execute("ALTER TABLE agent_versions ADD COLUMN concise boolean NOT NULL DEFAULT false;")


def downgrade() -> None:
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS concise;")
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS concise;")
