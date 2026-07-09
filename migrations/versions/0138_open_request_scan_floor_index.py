"""Add ``events_request_opened_seq_idx`` — the ``(session_id, seq)`` partial
index that lets the floor-bounded open-obligations anti-join (#1747) seek a
seq range instead of heap-filtering every ``request_opened`` row in
``[floor, head]``.

Companion to 0137 (the transactional ``sessions.open_request_scan_floor``
column add). Split into its own revision because mixing transactional DDL
with ``CREATE INDEX CONCURRENTLY`` in one revision is a partial-apply wedge
(see 0137's docstring) — precedent 0099 / 0128 / 0131 do *only*
``CONCURRENTLY`` per revision.

Why a NEW partial index rather than reusing the pre-existing non-partial
``events_session_seq_idx`` (mig 0080, ``(session_id, seq)`` over ALL event
kinds): floor-bounding against that index would already deliver
lifetime-independence (the acceptance win) with zero new indexes, but it
leaves the residual bounded scan at O(events-since-floor) — the non-partial
index returns every event kind in the seq range and the query heap-filters
for ``request_opened``. The target case this whole change exists for — a
re-invoked servicer holding one obligation open across many steps — is
exactly where ``events-since-floor >> request-edges-since-floor`` (thousands
of intervening non-``request_opened`` events between one ask and the next).
The new partial index tightens the residual to
O(request-edges-since-floor). Decision recorded in #1747: ship the partial
index; ``events_session_seq_idx`` is the documented zero-new-index fallback
if prod telemetry ever shows the residual immaterial.

Before building, DROP any invalid remnant first: an interrupted concurrent
build leaves ``pg_index.indisvalid = false``, and a bare
``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` then sees the *name* already
exists and silently skips it — leaving a permanently-unused invalid index
and a silent perf-nil (the exact hazard 0097's docstring calls out for its
own ``_new``-suffix dance). ``DROP INDEX CONCURRENTLY IF EXISTS`` is a
no-op when the index is absent or already valid-but-unused, and clears a
poisoned remnant when present.

Keeps 0099's ``events_request_opened_idx`` (session_id, (data->>'request_id'))
untouched — that index backs the point lookups (``get_request_caller`` et
al.), a different access pattern (equality on request_id, not a seq range).

Post-deploy, the ops-agent asserts ``indisvalid = true`` for this index (see
#1747 "Immune check") — a green migration is not proof the index is usable;
``pg_indexes``/``pg_index`` don't surface a failed CONCURRENTLY build as a
migration failure by themselves without that follow-up check.

Revision ID: 0138
Revises: 0137
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0138"
down_revision: str = "0137"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "events_request_opened_seq_idx"

_DROP_REMNANT = f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}"

_CREATE_INDEX = (
    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} "
    "ON events (session_id, seq) "
    "WHERE kind = 'lifecycle' AND data->>'event' = 'request_opened'"
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        # Fail-closed on a poisoned remnant, then build (see docstring).
        op.execute(_DROP_REMNANT)
        op.execute(_CREATE_INDEX)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
