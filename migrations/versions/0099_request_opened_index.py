"""Add the ``request_opened`` partial index — the trusted *ask* half of the
request edge (#1123).

Issue #1123 promotes the trusted invocation link to a first-class, typed
``request_opened`` lifecycle event appended by the launch-path creation
functions in the same transaction as the servicer they open. The open-request
set is re-derived as ``asked(request_opened) MINUS answered(request_response)``
— both typed lifecycle events. This migration adds the partial index that makes
the ``asked`` read a point lookup, mirroring ``events_request_response_idx``
(mig 0069), which already backs the ``answered`` read.

Purely additive — a single ``CREATE INDEX CONCURRENTLY``. No ``events.kind``
CHECK migration is needed: ``kind='lifecycle'`` is already legal (mig 0001).
Built via ``op.get_context().autocommit_block()`` so it never takes an ACCESS
EXCLUSIVE lock on the live-written ``events`` table (same pattern as
0069 / 0097). Safe in the post-deploy new-code/old-schema window: the new code's
``request_opened`` writes/reads work against the old schema (the index only
speeds the read up), and the index is invisible to the running container until
it completes.

Revision ID: 0099
Revises: 0098
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0099"
down_revision: str = "0098"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_request_opened_idx "
            "ON events (session_id, (data->>'request_id')) "
            "WHERE kind = 'lifecycle' AND data->>'event' = 'request_opened'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_request_opened_idx")
