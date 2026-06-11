"""Widen ``wf_run_events.type`` CHECK to include ``annotation``.

``log()`` and ``phase()`` become journaled progress events: a new ``annotation``
row in each run's journal, branch-local-keyed so the memo
``UNIQUE NULLS NOT DISTINCT (run_id, call_key, type)`` makes them emit-once across
replays. The inline CHECK from 0064 enumerated only the four capability/bookend
types, so it must be widened or every annotation insert raises.

Drop + re-add of the inline CHECK (PG-default name ``wf_run_events_type_check``
from 0064), exactly as 0074 widened ``wf_run_signals_kind_check``. Additive and
backward-compatible: existing rows are unaffected, and pre-deploy code never
writes the new type — so deploy ordering is unconstrained.

Revision ID: 0085
Revises: 0084
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0085"
down_revision: str = "0084"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_run_events DROP CONSTRAINT wf_run_events_type_check")
    op.execute(
        "ALTER TABLE wf_run_events ADD CONSTRAINT wf_run_events_type_check "
        "CHECK (type IN ('run_started','call_started','call_result','run_completed','annotation'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_run_events DROP CONSTRAINT wf_run_events_type_check")
    op.execute(
        "ALTER TABLE wf_run_events ADD CONSTRAINT wf_run_events_type_check "
        "CHECK (type IN ('run_started','call_started','call_result','run_completed'))"
    )
