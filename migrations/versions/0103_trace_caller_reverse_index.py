"""Add the reverse "children-of by caller" indexes — the trace walk's down-link
(#1149), shared with #1152's cancel-cascade.

The request edge is indexed for the *up* direction (``events_request_opened_idx``
/ mig 0099 keys ``(session_id, (data->>'request_id'))``; a trace instead walks
*down* — "every edge whose ``caller.id`` is this node." That reverse direction is
unindexed today (the ``wf_runs.caller`` JSONB column from mig 0101 is likewise
unindexed). #1149 builds it once and #1152 (cancel-cascade) reuses the same
children-of direction.

**Implementation fork (decided at PR time, per the lock-direction-defer-DDL
convention).** Two candidates were on the table: a JSONB **expression** index vs.
a promoted/generated ``caller_id`` column indexed plainly. This migration picks
the **expression index** — it is purely additive (no table rewrite, no
``ALTER TABLE … ADD COLUMN``), it mirrors the existing
``events_request_opened_idx`` expression-index pattern verbatim, and it keys the
*exact* predicate the trace query filters on
(``data->'caller'->>'kind'`` + ``data->'caller'->>'id'`` for events;
``caller->>'kind'`` + ``caller->>'id'`` for runs), so the
``EXPLAIN``-asserting integration test can pin index use to the query predicate.
The promoted-column variant (attractive for #1131's ``run_children_usage``
rewrite) is left as a follow-up that can swap the index out without changing the
query shape.

Both built via ``op.get_context().autocommit_block()`` so neither takes an ACCESS
EXCLUSIVE lock on the live-written ``events`` / ``wf_runs`` tables (same pattern
as 0069 / 0097 / 0099). Safe in the new-code/old-schema window: the trace read
works against the old schema (the index only speeds it up), and the index is
invisible to a running container until it completes.

Revision ID: 0103
Revises: 0102
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0103"
down_revision: str = "0102"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        # run/session → child session: the reverse lookup on a ``request_opened``
        # edge keyed on (caller.kind, caller.id). NOT id-alone — ``api`` edges
        # store ``caller.id = account_id``, so kind is part of the key.
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_request_opened_caller_idx "
            "ON events (account_id, (data->'caller'->>'kind'), (data->'caller'->>'id')) "
            "WHERE kind = 'lifecycle' AND data->>'event' = 'request_opened'"
        )
        # run → sub-run: the reverse lookup on ``wf_runs.caller`` (#1129).
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS wf_runs_caller_idx "
            "ON wf_runs (account_id, (caller->>'kind'), (caller->>'id'))"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_request_opened_caller_idx")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS wf_runs_caller_idx")
