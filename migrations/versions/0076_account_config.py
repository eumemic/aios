"""Accounts: add a typed per-account ``config`` JSONB bag.

The home for per-account settings (first item: ``timezone``, used to render the
per-message received-at envelope for the account's agents). Additive with a
``'{}'`` default, so every existing row is a valid empty config and no backfill
is needed. The ``accounts`` table is small and rarely written, so a plain
``ADD COLUMN`` (a metadata-only change in PG11+, no table rewrite) is fine — no
``CONCURRENTLY``.

Revision ID: 0076
Revises: 0075
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0076"
down_revision: str = "0075"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE accounts ADD COLUMN config jsonb NOT NULL DEFAULT '{}'::jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE accounts DROP COLUMN IF EXISTS config")
