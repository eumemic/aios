"""Add ``litellm_extra`` JSONB column to agents + agent_versions.

Lets an agent pin provider-specific LiteLLM kwargs (OpenRouter
``extra_body.provider.order``, Anthropic ``thinking``, OpenAI
``reasoning_effort``, raw sampling params, per-agent ``api_base``,
etc.) without stuffing them into the loosely-specified ``metadata``
column.  Default ``'{}'::jsonb`` so existing rows Just Work; changing
the column creates a new agent version (same as changing ``model``).

Revision ID: 0021
Revises: 0020
Create Date: 2026-04-21
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0021"
down_revision: str = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agents "
        "ADD COLUMN litellm_extra jsonb NOT NULL DEFAULT '{}'::jsonb;"
    )
    op.execute(
        "ALTER TABLE agent_versions "
        "ADD COLUMN litellm_extra jsonb NOT NULL DEFAULT '{}'::jsonb;"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS litellm_extra;")
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS litellm_extra;")
