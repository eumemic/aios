"""Partial index on the workflow child's request-response event.

A workflow agent child answers its request with a single ``workflow_child_done``
lifecycle event (see ``tools/workflow_completion.py`` /
``queries.write_response_if_absent``). ``queries.read_workflow_child_done`` reads
that one event on every parent run-step harvest to resolve the ``agent()`` call;
this partial index keeps that a point lookup instead of a per-wake history scan.
The same index serves ``write_response_if_absent``'s exactly-once absent-recheck.

Responses are rare relative to total events (one per finished child), so this is a
small partial index. Built with ``CREATE INDEX CONCURRENTLY`` (outside a
transaction via ``autocommit_block``) so it never takes an ACCESS EXCLUSIVE lock
on the live-written ``events`` table — same pattern as migrations 0023 / 0062 /
0065.

Revision ID: 0067
Revises: 0066
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0067"
down_revision: str = "0066"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_workflow_child_done_idx "
            "ON events (session_id) "
            "WHERE kind = 'lifecycle' AND data->>'event' = 'workflow_child_done'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_workflow_child_done_idx")
