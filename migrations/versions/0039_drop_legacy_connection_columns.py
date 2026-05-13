"""Drop the legacy ``connections.{session_id,session_template_id,tools}``
columns and the legacy ``connection_chat_sessions`` table.

Replacements:

* ``connections.session_id`` / ``session_template_id`` — projected from
  the active row of the ``bindings`` table via LEFT JOIN.
* ``connections.tools`` — superseded by ``connectors.tools_schema``
  (per connector type).
* ``connection_chat_sessions`` — superseded by ``chat_sessions``.

``CASCADE`` on the column drops sweeps along the partial index on
``session_id``, the FKs to ``sessions(id)`` / ``session_templates(id)``,
and the ``connections_one_mode_ck`` CHECK that gated mode-uniqueness on
the legacy shape.

Revision ID: 0039
Revises: 0038
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0039"
down_revision: str = "0038"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # One ALTER per relation: a single ACCESS EXCLUSIVE lock cycle on
    # ``connections`` for all three drops, instead of three serialized
    # waits during deploy.
    op.execute(
        "ALTER TABLE connections "
        "DROP COLUMN IF EXISTS session_id CASCADE, "
        "DROP COLUMN IF EXISTS session_template_id CASCADE, "
        "DROP COLUMN IF EXISTS tools"
    )
    op.execute("DROP TABLE IF EXISTS connection_chat_sessions")


def downgrade() -> None:
    # Reconstitutes the legacy shape but not the data.  Operators downgrading
    # past 0039 should restore from a snapshot rather than rely on this path.
    op.execute(
        """
        ALTER TABLE connections
            ADD COLUMN IF NOT EXISTS session_id          text REFERENCES sessions(id),
            ADD COLUMN IF NOT EXISTS session_template_id text REFERENCES session_templates(id),
            ADD COLUMN IF NOT EXISTS tools               jsonb NOT NULL DEFAULT '[]'::jsonb
        """
    )
    op.execute(
        """
        ALTER TABLE connections
            ADD CONSTRAINT connections_one_mode_ck CHECK (
                (session_id IS NULL AND session_template_id IS NULL)
                OR (session_id IS NOT NULL AND session_template_id IS NULL)
                OR (session_id IS NULL AND session_template_id IS NOT NULL)
            )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS connections_session_id_idx "
        "ON connections (session_id) WHERE archived_at IS NULL AND session_id IS NOT NULL"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS connection_chat_sessions (
            connection_id  text NOT NULL REFERENCES connections(id) ON DELETE CASCADE,
            chat_id        text NOT NULL,
            session_id     text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            created_at     timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (connection_id, chat_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS connection_chat_sessions_session_idx "
        "ON connection_chat_sessions (session_id)"
    )
