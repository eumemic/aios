"""Add per-session environment variables.

External orchestrators need to inject env vars (API URLs, run IDs, etc.)
into sandbox containers on a per-session basis.  The ``env`` column stores
a ``dict[str, str]`` as JSONB.  At container provisioning time, these are
merged with the environment-level ``config.env`` (session wins on key
collisions) and passed as ``--env`` flags to ``docker run``.

Revision ID: 0014
Revises: 0013
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0014"
down_revision: str = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN env jsonb NOT NULL DEFAULT '{}'::jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN env")
