"""Custom tools support + environment config.

Two independent changes in one migration:

1. Convert ``sessions.stop_reason`` from ``text`` to ``jsonb``. Custom tools
   need a structured stop_reason ``{"type": "requires_action", "event_ids":
   [...]}`` to tell clients which tool calls are pending. Existing string
   values (``'end_turn'``, ``'error'``, ``'interrupt'``) are migrated to
   ``{"type": "<value>"}``.

2. Add ``environments.config`` as ``jsonb``. Stores package lists and
   networking rules for container provisioning.

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Convert stop_reason from text to jsonb.
    op.execute(
        """
        ALTER TABLE sessions ALTER COLUMN stop_reason TYPE jsonb USING
            CASE
                WHEN stop_reason IS NULL THEN NULL
                ELSE jsonb_build_object('type', stop_reason)
            END;
        """
    )

    # 2. Add config column to environments.
    op.execute(
        "ALTER TABLE environments ADD COLUMN config jsonb NOT NULL DEFAULT '{}'::jsonb;"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE environments DROP COLUMN IF EXISTS config;")
    op.execute(
        """
        ALTER TABLE sessions ALTER COLUMN stop_reason TYPE text USING
            stop_reason->>'type';
        """
    )
