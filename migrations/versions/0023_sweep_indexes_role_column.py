"""Re-predicate the two sweep partial indexes from ``data->>'role'`` to the
``role`` column so the sweep queries can use the normalized column (promoted
in 0022) without losing index coverage.

Issue #140. The queries in ``sweep.py`` were reaching into JSONB with
``data->>'role'`` even after 0022 added the real ``role`` column, because
the partial indexes ``events_tool_result_idx`` and
``events_assistant_tool_calls_idx`` both had ``data->>'role' = <value>``
in their WHERE clauses. Any query that swapped to the column would lose
the index match. This migration updates both indexes to use ``role``,
unblocking the query rewrites in the same PR.

Run with ``CREATE INDEX CONCURRENTLY`` / ``DROP INDEX CONCURRENTLY`` so the
switchover never takes an ACCESS EXCLUSIVE lock on ``events`` — the table
is live-written by every session. The price is that we can't run inside
a transaction, so we disable alembic's implicit BEGIN via
``op.get_context().autocommit_block()``.

The ``role`` column was backfilled in 0022 from ``data->>'role'`` for every
message-kind row, so the set of rows covered by each partial index is
unchanged byte-for-byte.

Revision ID: 0023
Revises: 0022
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0023"
down_revision: str = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        # events_tool_result_idx: tool-result lookup by tool_call_id.
        op.execute(
            "CREATE INDEX CONCURRENTLY events_tool_result_idx_new "
            "ON events (session_id, (data->>'tool_call_id')) "
            "WHERE kind = 'message' AND role = 'tool'"
        )
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_tool_result_idx")
        op.execute("ALTER INDEX events_tool_result_idx_new RENAME TO events_tool_result_idx")

        # events_assistant_tool_calls_idx: scan for assistant messages with tool_calls.
        op.execute(
            "CREATE INDEX CONCURRENTLY events_assistant_tool_calls_idx_new "
            "ON events (session_id, seq) "
            "WHERE kind = 'message' AND role = 'assistant' "
            "AND data ? 'tool_calls'"
        )
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_assistant_tool_calls_idx")
        op.execute(
            "ALTER INDEX events_assistant_tool_calls_idx_new "
            "RENAME TO events_assistant_tool_calls_idx"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY events_tool_result_idx_old "
            "ON events (session_id, (data->>'tool_call_id')) "
            "WHERE kind = 'message' AND data->>'role' = 'tool'"
        )
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_tool_result_idx")
        op.execute("ALTER INDEX events_tool_result_idx_old RENAME TO events_tool_result_idx")

        op.execute(
            "CREATE INDEX CONCURRENTLY events_assistant_tool_calls_idx_old "
            "ON events (session_id, seq) "
            "WHERE kind = 'message' AND data->>'role' = 'assistant' "
            "AND data ? 'tool_calls'"
        )
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_assistant_tool_calls_idx")
        op.execute(
            "ALTER INDEX events_assistant_tool_calls_idx_old "
            "RENAME TO events_assistant_tool_calls_idx"
        )
