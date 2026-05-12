"""Drop ``sessions.spawned_from_connection_id`` (#328 PR 7).

The column was added in migration 0027 to express the per_chat lineage
("which connection spawned this session?") and double-duty as the
focal-lock signal for ``switch_channel``. PR 3 migration 0034 added a
dedicated ``sessions.focal_locked`` boolean and backfilled it from the
non-NULL set of ``spawned_from_connection_id``, eliminating the
double-duty. PR 7 finishes the job:

* The per-chat lineage now lives entirely in the ``chat_sessions``
  table (``(connection_id, session_id)`` rows).  Every reader that
  consulted ``spawned_from_connection_id`` (``is_session_bound_to_connection``,
  ``list_connection_tools_for_session``, ``_list_bound_connection_ids``,
  ``list_pending_calls_for_connection``) has been rewritten to walk
  ``bindings`` + ``chat_sessions``.
* Writers (``sessions_service.create_session``, the per_chat resolver)
  no longer pass ``spawned_from_connection_id``; per_chat-spawned
  sessions set ``focal_locked=True`` directly.

Revision ID: 0036
Revises: 0035
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0036"
down_revision: str = "0035"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS spawned_from_connection_id")


def downgrade() -> None:
    # Reconstituting the column is lossy — the lineage data is gone.
    # Provided so ``alembic downgrade`` doesn't error; not a real
    # rollback path (the FK to ``connections`` cannot be restored
    # without re-running the original 0027 backfill against a
    # connections snapshot we no longer have).
    op.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS "
        "spawned_from_connection_id text REFERENCES connections(id)"
    )
