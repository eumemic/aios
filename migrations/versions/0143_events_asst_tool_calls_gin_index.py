"""Add the assistant tool-call containment GIN index (#1737).

The sweep batch filter probes ``data->'tool_calls' @> ANY($::jsonb[])`` while
scoped to assistant messages.  Previously PostgreSQL used the session/message
btree and filter-detoasted every message in a large session.  This expression
GIN index lets PostgreSQL answer the containment condition directly without an
application query change.

The partial predicate intentionally does *not* include ``data ? 'tool_calls'``.
Although rows without that key add no GIN keys, the extra predicate would make
the index unusable: PostgreSQL cannot prove that the unmodified containment
probe implies the key-existence predicate.

Built concurrently in an autocommit block because ``events`` is hot and live
written.  If a concurrent build is interrupted, it can leave an INVALID index
that ``IF NOT EXISTS`` will silently retain.  Before retrying, operators must
remove that remnant with::

    DROP INDEX CONCURRENTLY IF EXISTS events_asst_tool_calls_gin_idx;

Revision ID: 0143
Revises: 0142
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0143"
down_revision: str = "0142"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "events_asst_tool_calls_gin_idx"
CREATE_INDEX = (
    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} "
    "ON events USING gin ((data->'tool_calls') jsonb_path_ops) "
    "WHERE kind = 'message' AND role = 'assistant'"
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(CREATE_INDEX)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
