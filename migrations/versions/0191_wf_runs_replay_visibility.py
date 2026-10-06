"""Hide runs that can read an agent's past requests from every agent (#2475).

Revision ID: 0191
Revises: 0190

A run that declares a replay tool (``sample_requests`` or ``get_request``) reads
other sessions' requests and journals what it reads. Only the operator may see
it, so the visibility trigger (0185) stamps it ``session``. Such a run is
operator-launched and has no launching session, and a ``session`` run is visible
only to its launching session, so no agent's run-read tools can see it. Its
sub-runs inherit ``session`` through the trigger's existing parent arm, and their
launching session is copied from the parent, so they are hidden too.

The trigger stays the column's only writer. An application image from before
this revision can't insert such a run: it rejects the tool types when it
validates a workflow, and raises when it reads one. No existing row declares
them, so there is nothing to backfill.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0191"
down_revision: str | None = "0190"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _replace_function(replay_arm: str) -> None:
    op.execute(rf"""
        CREATE OR REPLACE FUNCTION _aios_stamp_wf_run_visibility() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            NEW.visibility := CASE
                {replay_arm}
                WHEN NEW.caller->>'purpose' = 'model_dispatch' THEN 'session'
                WHEN NEW.parent_run_id IS NOT NULL THEN
                    (SELECT visibility FROM wf_runs WHERE id = NEW.parent_run_id)
                ELSE 'account'
            END;
            RETURN NEW;
        END
        $$
    """)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    _replace_function(
        """WHEN NEW.tools @> '[{"type": "sample_requests"}]'::jsonb
                  OR NEW.tools @> '[{"type": "get_request"}]'::jsonb THEN 'session'"""
    )


def downgrade() -> None:
    _replace_function("")
