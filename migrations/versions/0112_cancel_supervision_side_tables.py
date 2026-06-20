"""Cancel supervision side-tables (cancel-design §0/§9, delta 1+2).

The durable substrate the recursive ``cancel_invocation`` cascade is built on. Two
additive tables, zero row rewrites — both are **side-tables** the supervision tree
writes to, NEVER another node's event log/journal (the single-writer invariant: a
node's own step under its own lock is the only writer of its log).

- ``cancel_intents`` — the operator-intent **tombstone**, one row per
  ``cancel_invocation`` call, keyed by the cancelled edge handle
  ``(servicer_kind, servicer_id, request_id)``. Written first, independent of
  cascade progress. Carries the §9 monotone ``outstanding`` quiescence counter
  (seed 1; each node adjusts it in its own terminal/withdraw txn) and
  ``quiesced_at`` (set the instant ``outstanding`` hits 0) — a side-table UPDATE,
  never a journal append, so it is invariant-safe.

- ``session_cancel_markers`` — the session-side **exit-marker** (the run side
  reuses ``wf_run_signals kind='cancel'`` verbatim, so no run table is added).
  Keyed by the target edge ``(session_id, request_id)`` with the cascade's
  ``ON CONFLICT DO NOTHING`` idempotency. A marker is a durable exit signal the
  target session's own step harvests under its lock; ``harvested_at`` flips once
  it has, so the sweep wakes only still-unharvested markers (C2).

``downgrade()`` drops both. No data migration — both tables start empty.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0112"
down_revision: str = "0111"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE cancel_intents (
            servicer_kind text        NOT NULL CHECK (servicer_kind IN ('session','run')),
            servicer_id   text        NOT NULL,
            request_id    text        NOT NULL,
            -- The polymorphic ``(servicer_kind, servicer_id)`` can't FK a single table, so
            -- ``account_id`` carries the only cascade path: a deleted account takes its
            -- tombstones with it (no orphan leak). The marker table reaches deletion
            -- transitively via its ``session_id`` cascade instead.
            account_id    text        NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            -- The §9 monotone quiescence counter. NOT CHECK-bounded: it is a best-effort
            -- hint backed by the 30s reconcile re-derivation, so a transient miscount must
            -- never crash the node's own terminal transaction that adjusts it.
            outstanding   integer     NOT NULL DEFAULT 1,
            quiesced_at   timestamptz,
            created_at    timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (servicer_kind, servicer_id, request_id)
        )
    """)
    # Account-scoped reconcile/quiescence sweeps over still-open intents.
    op.execute(
        "CREATE INDEX cancel_intents_open ON cancel_intents (account_id) WHERE quiesced_at IS NULL"
    )

    op.execute(r"""
        CREATE TABLE session_cancel_markers (
            session_id   text        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            request_id   text        NOT NULL,
            account_id   text        NOT NULL,
            harvested_at timestamptz,
            created_at   timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, request_id)
        )
    """)
    # The C2 sweep clause selects sessions with an unharvested marker; this partial
    # index keeps that wake-scan cheap (one indexed row per pending exit).
    op.execute(
        "CREATE INDEX session_cancel_markers_unharvested "
        "ON session_cancel_markers (session_id) WHERE harvested_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS session_cancel_markers")
    op.execute("DROP TABLE IF EXISTS cancel_intents")
