"""Workflows: add an optional ``description`` to the definition.

The agent-parity blurb. ``agents`` already carry a ``description`` (surfaced in
the console list/search/detail); workflows did not. Additive + nullable, so it is
backward-compatible with every existing row and needs no backfill. The
``workflows`` table is small and rarely written, so a plain ``ADD COLUMN`` (a
metadata-only change in PG11+, no table rewrite) is fine — no ``CONCURRENTLY``.

Revision ID: 0071
Revises: 0070
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0071"
down_revision: str = "0070"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE workflows ADD COLUMN description text")


def downgrade() -> None:
    op.execute("ALTER TABLE workflows DROP COLUMN IF EXISTS description")
