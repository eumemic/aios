"""Widen ``wf_run_events.type`` CHECK to include ``request_response`` (#1126).

A run completing *in service of a request* emits a ``request_response`` journal
event at its terminal ``_complete_run`` chokepoint — the run-side mirror of the
session ``respond_to_request`` writer — keyed on the inbound ``request_id`` via
``call_key`` so the existing ``UNIQUE NULLS NOT DISTINCT (run_id, call_key,
type)`` index latches it exactly-once alongside ``run_completed`` (replay /
procrastinate dual execution never double-emits). The ``wf_run_events.type``
CHECK predates this event type, so it must be widened or every emission raises.

Purely additive to the type-set (the same shape as 0088 widening it for
``frontier_deferred``); no data is rewritten. Safe in the post-deploy
new-code/old-schema window because the new code only ever *inserts*
``request_response`` rows once this CHECK admits them — until the migration
completes the run-caller path that produces them is not yet exercised against
the old schema.

Revision ID: 0102
Revises: 0101
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0102"
down_revision: str = "0101"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_run_events DROP CONSTRAINT wf_run_events_type_check")
    op.execute(
        "ALTER TABLE wf_run_events ADD CONSTRAINT wf_run_events_type_check "
        "CHECK (type IN ('run_started','call_started','call_result','run_completed',"
        "'annotation','frontier_deferred','request_response'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_run_events DROP CONSTRAINT wf_run_events_type_check")
    op.execute(
        "ALTER TABLE wf_run_events ADD CONSTRAINT wf_run_events_type_check "
        "CHECK (type IN ('run_started','call_started','call_result','run_completed',"
        "'annotation','frontier_deferred'))"
    )
