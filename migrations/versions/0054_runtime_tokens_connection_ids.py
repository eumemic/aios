"""Add nullable ``connection_ids text[]`` to ``runtime_tokens`` (#350).

The optional allowlist scope: when ``connection_ids`` is non-NULL on
a runtime token, the bearer can only see / operate on the listed
connection IDs.  ``NULL`` means "unscoped" — pre-#350 behaviour where
the token sees every connection of its connector type.

Precedent for ``text[]`` (vs. jsonb): ``session_templates.vault_ids``
and ``session_templates.memory_store_ids``.  Backwards-safe because
the column is nullable and has no default — pre-migration rows
become ``NULL`` (unscoped).

Revision ID: 0054
Revises: 0053
Create Date: 2026-05-20
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0054"
down_revision: str = "0053"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE runtime_tokens ADD COLUMN connection_ids text[]")


def downgrade() -> None:
    op.execute("ALTER TABLE runtime_tokens DROP COLUMN IF EXISTS connection_ids")
