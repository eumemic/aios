"""Widen ``wf_run_events.type`` CHECK to include ``frontier_deferred``.

Per-run wave admission (#784) bounds the number of concurrently in-flight
``agent()`` children. A frontier the wave gate defers is JOURNALED — a new
``frontier_deferred`` row in the run's journal — so the replay-determinism
guard can see deferred-but-never-started keys (rather than them looking like
a vanished inflight call). The inline CHECK from 0064 (widened to
``annotation`` in 0085) enumerated only the prior types, so it must be widened
again or every ``frontier_deferred`` insert raises.

Drop + re-add of the inline CHECK (PG-default name ``wf_run_events_type_check``
from 0064), exactly as 0085 widened it to ``annotation``. Additive and
backward-compatible: existing rows are unaffected, and pre-deploy code never
writes the new type — so deploy ordering is unconstrained.

Revision ID: 0087
Revises: 0086
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0087"
down_revision: str = "0086"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_run_events DROP CONSTRAINT wf_run_events_type_check")
    op.execute(
        "ALTER TABLE wf_run_events ADD CONSTRAINT wf_run_events_type_check "
        "CHECK (type IN ('run_started','call_started','call_result','run_completed',"
        "'annotation','frontier_deferred'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_run_events DROP CONSTRAINT wf_run_events_type_check")
    op.execute(
        "ALTER TABLE wf_run_events ADD CONSTRAINT wf_run_events_type_check "
        "CHECK (type IN ('run_started','call_started','call_result','run_completed','annotation'))"
    )
