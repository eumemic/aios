"""Backfill ``account_id`` on every reserved resource table and ``SET NOT NULL``.

PR 4's migration 0043 renamed ``user_id`` → ``account_id`` and added a
nullable FK on every reserved table. PR 5a threaded real account values
to every call site; PR 5b added ``account_id`` to every resource-table
``INSERT``. This migration is the third leg: backfill any pre-existing
``NULL`` rows to the root account, then ``SET NOT NULL`` so the column
becomes a hard tenancy invariant.

``connectors`` is intentionally excluded — it's a global per-type registry
(signal, telegram, …), not a tenant-scoped resource. Its ``account_id``
column stays nullable so the global upsert pattern keeps working.

Empty-database guard: tests run ``alembic upgrade head`` before any
bootstrap. The ``UPDATE`` then matches zero rows (no NULLs to backfill),
the SELECT subquery returns NULL but no row needed it, and ``SET NOT
NULL`` succeeds because the table is empty. On a populated production
database, the operator must have already bootstrapped a root account
(via ``POST /v1/accounts/bootstrap``) before running this migration —
otherwise the backfill leaves NULLs and ``SET NOT NULL`` fails loudly,
which is the right behavior.

Revision ID: 0044
Revises: 0043
Create Date: 2026-05-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0044"
down_revision: str = "0043"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TENANT_SCOPED_TABLES: Sequence[str] = (
    "bindings",
    "chat_sessions",
    "routing_rules",
    "runtimes",
    "runtime_tokens",
    "inbound_acks",
    "agent_versions",
    "agents",
    "connections",
    "credentials",
    "environments",
    "events",
    "files",
    "memories",
    "memory_stores",
    "memory_versions",
    "session_github_repositories",
    "session_memory_stores",
    "session_templates",
    "session_vaults",
    "skill_versions",
    "skills",
    "vault_credentials",
    "vaults",
    "sessions",
)


def upgrade() -> None:
    for table in _TENANT_SCOPED_TABLES:
        op.execute(
            f"""
            UPDATE {table}
               SET account_id = (
                 SELECT id FROM accounts
                  WHERE parent_account_id IS NULL AND archived_at IS NULL
                  LIMIT 1
               )
             WHERE account_id IS NULL
            """
        )
        op.execute(f"ALTER TABLE {table} ALTER COLUMN account_id SET NOT NULL")


def downgrade() -> None:
    for table in reversed(_TENANT_SCOPED_TABLES):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN account_id DROP NOT NULL")
