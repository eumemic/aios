"""Connector capability descriptor — a ``tools_schema`` sibling on the
``connectors`` catalog row (#1381).

Purely-additive ``ADD COLUMN connectors.capabilities jsonb NOT NULL
DEFAULT '{}'::jsonb`` — the exact same shape as the ``tools_schema`` sibling
seeded at migration 0033 (``DEFAULT '[]'``).  The ``'{}'`` default means every
existing connector row reads as "no declared capabilities" (all sub-descriptors
absent) — the conservative/plain-text rendering floor.  No backfill needed; the
default IS the floor.

Safe across the new-code/old-schema deploy window per CLAUDE.md: an additive
``ADD COLUMN`` is invisible to the running container until complete, old code
never reads the new column, and there is no expand/contract.  Reversible via
``DROP COLUMN``.  (No ``@dataclass`` in the migration body — alembic loads
versions under synthetic module names and a dataclass there crashes.)

Revision ID: 0113
Revises: 0112
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0113"
down_revision: str = "0112"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE connectors "
        "ADD COLUMN capabilities jsonb NOT NULL DEFAULT '{}'::jsonb"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE connectors DROP COLUMN capabilities")
