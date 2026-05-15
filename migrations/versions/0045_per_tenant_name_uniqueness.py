"""Re-scope per-tenant resource-name uniqueness from global to per-account.

Pre-multi-tenancy, names like ``agents.name``, ``environments.name``,
``credentials.name``, and ``session_templates.name`` had global unique
indexes scoped to ``archived_at IS NULL``. Now that every row belongs to
an ``account``, two different tenants minting an agent named "default"
should both succeed — the constraint should be ``(account_id, name)``
not just ``(name)``.

This migration drops the global partial indexes and recreates them
scoped to ``(account_id, name)``. The ``WHERE archived_at IS NULL``
predicate is preserved so archived rows don't claim the namespace.

Revision ID: 0045
Revises: 0044
Create Date: 2026-05-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0045"
down_revision: str = "0044"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# (table, index_name, column-set used in the v1 global index).
_INDEXES: Sequence[tuple[str, str, str]] = (
    ("agents", "agents_name_uniq", "name"),
    ("environments", "environments_name_uniq", "name"),
    ("credentials", "credentials_name_uniq", "name"),
    ("session_templates", "session_templates_name_uniq", "name"),
)


def upgrade() -> None:
    for table, index_name, column in _INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {index_name}")
        op.execute(
            f"CREATE UNIQUE INDEX {index_name} "
            f"ON {table} (account_id, {column}) WHERE archived_at IS NULL"
        )


def downgrade() -> None:
    for table, index_name, column in _INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {index_name}")
        op.execute(
            f"CREATE UNIQUE INDEX {index_name} "
            f"ON {table} ({column}) WHERE archived_at IS NULL"
        )
