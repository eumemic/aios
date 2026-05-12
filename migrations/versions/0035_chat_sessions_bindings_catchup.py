"""Catch-up backfill of bindings + chat_sessions from legacy shape (#328 PR 7).

Migration 0033 seeded the new subsystem tables from the live data at PR 2
deploy time. Between PR 2 and PR 7 the legacy in-place columns
(``connections.session_id``/``session_template_id``) and the legacy
``connection_chat_sessions`` ledger were still authoritative, so any
attaches / configures / chat-binds during that window populated only the
legacy shape. PR 7 cuts every reader over to ``bindings`` / ``chat_sessions``;
this migration copies the post-0033 writes into the new tables so the
cutover doesn't lose live routing data.

Idempotent: both backfills use ``ON CONFLICT DO NOTHING`` against the
new tables' unique constraints.

Revision ID: 0035
Revises: 0034
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0035"
down_revision: str = "0034"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Catch up the chat ledger first — per_chat-spawned rows reference
    # the legacy ``connections.session_id`` / ``session_template_id``
    # columns only transitively.  Idempotent via the PK
    # ``(connection_id, chat_id)``.
    op.execute(
        """
        INSERT INTO chat_sessions (connection_id, chat_id, session_id, created_at)
        SELECT connection_id, chat_id, session_id, created_at
          FROM connection_chat_sessions
            ON CONFLICT (connection_id, chat_id) DO NOTHING
        """
    )

    # Catch up bindings from any active connection whose legacy
    # ``session_id`` / ``session_template_id`` is populated but which
    # has no active binding row yet.  The partial-unique index on
    # ``bindings (connection_id) WHERE archived_at IS NULL`` guarantees
    # the NOT EXISTS guard is race-free under serializable isolation
    # (migrations run in a single transaction).
    op.execute(
        """
        INSERT INTO bindings (id, connection_id, mode, session_id,
                              session_template_id, created_at)
        SELECT 'bnd_' || REPLACE(gen_random_uuid()::text, '-', ''),
               c.id,
               CASE WHEN c.session_id IS NOT NULL THEN 'single_session'
                    ELSE 'per_chat' END,
               c.session_id,
               c.session_template_id,
               COALESCE(c.attached_at, c.created_at)
          FROM connections c
         WHERE c.archived_at IS NULL
           AND (c.session_id IS NOT NULL OR c.session_template_id IS NOT NULL)
           AND NOT EXISTS (
               SELECT 1 FROM bindings b
                WHERE b.connection_id = c.id
                  AND b.archived_at IS NULL
           )
        """
    )


def downgrade() -> None:
    # Data-only migration; downgrade is a no-op (the inserted rows are
    # valid against the pre-0035 schema and can stay in place).
    pass
