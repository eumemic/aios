"""Backfill ``account_id`` nulls to the singular active root and set
NOT NULL on every resource table.

PR 4 (migration 0043) renamed ``user_id`` → ``account_id`` and added the
FK to ``accounts(id)`` but kept the column nullable to give application
code time to start populating it. PR 5 (this migration) is the second
half: backfill the rows the application hasn't tagged yet, then lock
the column with NOT NULL.

On a fresh install: backfill is a no-op (zero rows in resource tables),
SET NOT NULL is vacuously satisfied.

On an existing single-tenant deployment with data: the operator must
have called ``POST /v1/accounts/bootstrap`` before applying migrations
(otherwise the backfill subquery returns NULL and SET NOT NULL fails
with a clear error pointing at the unbackfilled rows).

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


_TABLES: Sequence[str] = (
    "connectors",
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
    for table in _TABLES:
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
    for table in reversed(_TABLES):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN account_id DROP NOT NULL")
