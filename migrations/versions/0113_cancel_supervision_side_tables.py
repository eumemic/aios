"""Cancel supervision side-table (cancel-design §2).

The durable substrate the recursive ``cancel_invocation`` cascade is built on: one
additive table, zero row rewrites — a **side-table** the supervision tree writes to,
NEVER another node's event log/journal (the single-writer invariant: a node's own step
under its own lock is the only writer of its log).

- ``session_cancel_markers`` — the session-side **exit-marker** (the run side
  reuses ``wf_run_signals kind='cancel'`` verbatim, so no run table is added).
  Keyed by the target edge ``(session_id, request_id)`` with the cascade's
  ``ON CONFLICT DO NOTHING`` idempotency. A marker is a durable exit signal the
  target session's own step harvests under its lock; ``harvested_at`` flips once
  it has, so the sweep wakes only still-unharvested markers (C2).

(The §9 ``cancel_intents`` tombstone + quiescence counter land WITH their driver —
the run-down cascade + quiescence accounting, #788/#1152 — not ahead of it.)

``downgrade()`` drops it. No data migration — the table starts empty.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0113"
down_revision: str = "0112"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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
