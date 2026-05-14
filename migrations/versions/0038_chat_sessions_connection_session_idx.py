"""Index ``chat_sessions(connection_id, session_id)`` for hot-path
lineage lookups (#328 PR 7).

Three per-step / per-tool-result queries scan the predicate
``EXISTS (SELECT 1 FROM chat_sessions cs WHERE cs.connection_id = $X
AND cs.session_id = $Y)`` — see
``_session_bound_to_connection_predicate`` in ``aios.db.queries``. The
table's PK is ``(connection_id, chat_id)`` (good for ledger lookups by
chat) and the only secondary index is ``(session_id)`` (good for fan-out
on the runtime SSE), so the exact-match by ``(connection_id, session_id)``
fell to a btree scan + filter.  This partial-data table stays small,
but the lookup is on the hot per-step path.

Revision ID: 0038
Revises: 0037
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0038"
down_revision: str = "0037"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS chat_sessions_connection_session_idx "
        "ON chat_sessions (connection_id, session_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chat_sessions_connection_session_idx")
