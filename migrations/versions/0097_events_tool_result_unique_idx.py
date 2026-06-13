"""Promote ``events_tool_result_idx`` to a partial UNIQUE index — the
structural floor under the manual tool-result dedup.

Issue #1082. The invariant *at most one ``role='tool'`` event per
``(session_id, tool_call_id)``* is enforced today by a manual
read-check-write dedup replicated at multiple appender sites
(``harness/tool_dispatch.py``, ``services/sessions.py``,
``db/queries/events.py:find_tool_result_event``). There is no DB
constraint: ``events_tool_result_idx`` (migrations 0011 → 0023) covers
``(session_id, (data->>'tool_call_id')) WHERE kind='message' AND
role='tool'`` but is **non-unique**, so a genuine concurrency race
(an ``always_allow`` builtin's operator-deny committing while the worker
tool task is in flight; the late worker append clobbering the deny) can
land a SECOND tool-role row — reachable, with a production incident on
record (#550, #890/#841).

This migration makes the duplicate **row** unrepresentable. Because
``append_event`` allocates the per-session seq and INSERTs in the
**same** ``conn.transaction()`` (``db/queries/events.py``), a
``UniqueViolation`` aborts the tx and rolls back the seq increment with
it — the gapless-seq invariant is preserved and the duplicate row never
lands. The catch handlers at the appender sites still re-read + classify
(idempotent-retry vs deny-after-success ``ConflictError``) and apply the
``open_tool_call_count`` compensation; this index is the floor under that
logic, not a replacement for it.

One-time pre-build dedup backfill: keep the **lowest-seq** row of any
pre-existing duplicate ``(session_id, data->>'tool_call_id')`` group.
``append_event`` allocates ``seq`` via ``UPDATE sessions SET
last_event_seq = last_event_seq + 1 ... RETURNING last_event_seq`` in the
same transaction as the INSERT, so ``seq`` is strictly monotonic per
session in insert order: the lowest-seq row is the first-inserted (the
winning/first tool result) — the same row the read-check-write dedup
would have preserved. We delete the rest, then build the UNIQUE index.

Built with ``CREATE UNIQUE INDEX CONCURRENTLY`` outside a transaction via
``op.get_context().autocommit_block()`` so it never takes an ACCESS
EXCLUSIVE lock on the live-written ``events`` table — same pattern as
migrations 0023 / 0065. If the build fails mid-flight (e.g. a duplicate
slipped in after the backfill), the partially-built ``*_new`` index
lingers as INVALID; drop it manually (``DROP INDEX CONCURRENTLY IF EXISTS
events_tool_result_idx_new``) before retrying.

Revision ID: 0097
Revises: 0095
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0097"
down_revision: str = "0095"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # One-time pre-build dedup backfill: keep the lowest-seq row of any
    # pre-existing duplicate group. Runs inside the migration's implicit
    # transaction (a normal DELETE; no concurrent-index concern), BEFORE
    # the autocommit_block that builds the UNIQUE index. Without this the
    # CONCURRENTLY build would fail on any deployment that already carries
    # a duplicate pair from the historical race.
    op.execute(
        """
        DELETE FROM events e
         WHERE e.kind = 'message'
           AND e.role = 'tool'
           AND e.seq > (
               SELECT MIN(e2.seq)
                 FROM events e2
                WHERE e2.session_id = e.session_id
                  AND e2.kind = 'message'
                  AND e2.role = 'tool'
                  AND e2.data->>'tool_call_id' = e.data->>'tool_call_id'
           )
        """
    )

    with op.get_context().autocommit_block():
        op.execute(
            "CREATE UNIQUE INDEX CONCURRENTLY events_tool_result_idx_new "
            "ON events (session_id, (data->>'tool_call_id')) "
            "WHERE kind = 'message' AND role = 'tool'"
        )
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_tool_result_idx")
        op.execute("ALTER INDEX events_tool_result_idx_new RENAME TO events_tool_result_idx")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY events_tool_result_idx_old "
            "ON events (session_id, (data->>'tool_call_id')) "
            "WHERE kind = 'message' AND role = 'tool'"
        )
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_tool_result_idx")
        op.execute("ALTER INDEX events_tool_result_idx_old RENAME TO events_tool_result_idx")
