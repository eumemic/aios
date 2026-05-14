"""Rename ``user_id`` → ``account_id`` on every reserved table and add a
nullable FK to ``accounts(id)``.

This is the first half of activating the multi-tenancy schema:

* Rename matches the conceptual model (``accounts`` is the tenant table
  per memory, not ``users``).
* FK to ``accounts(id) ON DELETE RESTRICT`` blocks accidental account
  deletion that would orphan resources.
* Column stays nullable for now — backfill + ``SET NOT NULL`` + adding
  ``WHERE account_id = $X`` clauses to every read/list/update query
  land in a follow-up migration so this one stays atomic and reviewable.

``sessions.owner_id`` is dropped here — it was a placeholder for what is
now ``account_id`` and never read by any code.

Revision ID: 0043
Revises: 0042
Create Date: 2026-05-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0043"
down_revision: str = "0042"
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
        op.execute(f"ALTER TABLE {table} RENAME COLUMN user_id TO account_id")
        op.execute(
            f"""
            ALTER TABLE {table}
              ADD CONSTRAINT {table}_account_id_fk
              FOREIGN KEY (account_id) REFERENCES accounts(id)
              ON DELETE RESTRICT
            """
        )
    op.execute("ALTER TABLE sessions DROP COLUMN owner_id")


def downgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN owner_id text")
    for table in reversed(_TABLES):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_account_id_fk")
        op.execute(f"ALTER TABLE {table} RENAME COLUMN account_id TO user_id")
