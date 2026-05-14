"""Core schema additions: sessions.owner_id, focal_locked, user_id reservations (#328 PR 3/8).

Adds the core-side columns the connector subsystem rework relies on:

* ``sessions.owner_id text NULL`` — opaque to core. The subsystem
  chooses the value; today it's a constant string (no principals table
  per #328's PR 2 design sign-off), in the future it carries
  multi-tenant ownership semantics.
* ``sessions.focal_locked boolean NOT NULL DEFAULT FALSE`` — the
  ``switch_channel`` tool's gate. Replaces the
  ``spawned_from_connection_id IS NOT NULL`` check that lived in the
  tool body. Backfilled here so the invariant survives the migration
  without any code-side coordination.
* ``user_id text NULL`` on every existing core user-data table.
  Reserved for the future multi-tenant migration — no enforcement,
  no filtering, no query changes. PR 2 already reserved this on the
  new subsystem tables; this migration completes the coverage.

``spawned_from_connection_id`` stays for now — drops in PR 7 alongside
its dependent queries.

Revision ID: 0034
Revises: 0033
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0034"
down_revision: str = "0033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Existing user-data tables that should reserve a nullable ``user_id``.
# Excludes tables already covered by PR 2 (bindings/connectors/etc) and
# tables on the chopping block (connector_tokens, connector_inbound_acks,
# connection_chat_sessions all drop in PR 7/8).
_USER_ID_TABLES: Sequence[str] = (
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
)


def upgrade() -> None:
    # ---- sessions: owner_id, focal_locked, user_id -----------------------
    op.execute(
        """
        ALTER TABLE sessions
          ADD COLUMN owner_id     text,
          ADD COLUMN focal_locked boolean NOT NULL DEFAULT FALSE,
          ADD COLUMN user_id      text
        """
    )

    # focal_locked backfill: preserve today's "per_chat-spawned sessions
    # can't switch focal" invariant after the switch_channel tool starts
    # reading focal_locked instead of spawned_from_connection_id.
    op.execute(
        """
        UPDATE sessions
           SET focal_locked = TRUE
         WHERE spawned_from_connection_id IS NOT NULL
        """
    )

    # ---- user_id reservation on every other user-data table --------------
    for table in _USER_ID_TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN user_id text")


def downgrade() -> None:
    for table in reversed(_USER_ID_TABLES):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS user_id")
    op.execute(
        """
        ALTER TABLE sessions
          DROP COLUMN IF EXISTS user_id,
          DROP COLUMN IF EXISTS focal_locked,
          DROP COLUMN IF EXISTS owner_id
        """
    )
