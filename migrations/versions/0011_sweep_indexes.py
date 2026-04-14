"""Partial indexes for the sweep function.

Two indexes optimize the sweep's ghost detection and inference queries:

1. ``events_assistant_tool_calls_idx`` — fast lookup of assistant messages
   with tool_calls, scoped by session. Used by ghost repair to find
   tool_call_ids that may be missing results.

2. ``events_tool_result_idx`` — fast lookup of tool results by
   tool_call_id within a session. Used by ghost repair to check whether
   a tool_call_id has a matching result, and by the inference detection
   query to verify batch completion.

Revision ID: 0011
Revises: 0010
"""

from alembic import op

revision = "0011"
down_revision = "0010"


def upgrade() -> None:
    op.execute(
        "CREATE INDEX events_assistant_tool_calls_idx "
        "ON events (session_id, seq) "
        "WHERE kind = 'message' "
        "AND data->>'role' = 'assistant' "
        "AND data ? 'tool_calls';"
    )
    op.execute(
        "CREATE INDEX events_tool_result_idx "
        "ON events (session_id, (data->>'tool_call_id')) "
        "WHERE kind = 'message' "
        "AND data->>'role' = 'tool';"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS events_tool_result_idx;")
    op.execute("DROP INDEX IF EXISTS events_assistant_tool_calls_idx;")
