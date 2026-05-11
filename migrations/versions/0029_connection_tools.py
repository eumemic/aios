"""Add tools jsonb column to connections.

Per #301: a connection can declare model-facing custom tools alongside
its account binding.  When a session has connections attached
(``single_session`` mode) or was spawned from a connection (``per_chat``
mode), those tools are sourced into the model's tool list as
``type="custom"``.  The model calls them, the session parks in
``requires_action``, the connector executes externally and POSTs the
result back via ``/v1/sessions/:id/tool-results``.

Default ``[]``::jsonb — existing connections are tool-less and behave
exactly as before.

Revision ID: 0029
Revises: 0028
Create Date: 2026-05-08
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0029"
down_revision: str = "0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE connections ADD COLUMN tools jsonb NOT NULL DEFAULT '[]'::jsonb")
    # Hot-path index: ``compute_step_prelude`` calls
    # ``list_connection_tools_for_session`` on every step.  The first
    # WHERE branch is ``c.session_id = $1`` (single_session attach) —
    # without an index the planner sequence-scans connections every step.
    # Partial index since archived connections are excluded by the same
    # query, and only single_session rows have ``session_id`` populated.
    op.execute(
        "CREATE INDEX connections_session_id_idx "
        "ON connections (session_id) WHERE archived_at IS NULL AND session_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS connections_session_id_idx")
    op.execute("ALTER TABLE connections DROP COLUMN IF EXISTS tools")
