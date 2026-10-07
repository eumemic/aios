"""A run an operator re-roots at an agent version acts for that agent (#2476 F3).

Revision ID: 0193
Revises: 0192

``invoke_workflow(..., as_agent=...)`` (#2472) clamps an operator run's sub-run to an
agent version's surface, so an eval arm runs with the authority the agent would give
it. Its principal was the parent's, ``operator``, so code the agent wrote (a
candidate workflow) passed every operator-only check: binding ``agent()`` children to
``workflow:`` models, the replay tools, ``as_agent`` itself. Such a run, and every run
beneath it, now has principal ``agent``, which fails those checks the way a
``session`` run does.

The principal trigger (0184) stays the column's only writer: it stamps ``agent`` when
the row has ``as_agent_id``, and a sub-run still inherits its parent's principal. The
backfill re-stamps the existing ``as_agent`` runs and their sub-runs, which only exist
where a pre-release eval ran.

An application image from before this revision can't read a run with principal
``agent``, and its ``create_run`` asserts an ``as_agent`` sub-run got its parent's
principal. Both only touch eval runs, which don't run before this deploys.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0193"
down_revision: str | None = "0192"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _replace_function(agent_arm: str) -> None:
    op.execute(rf"""
        CREATE OR REPLACE FUNCTION _aios_stamp_wf_run_principal() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            NEW.principal := CASE
                WHEN NEW.launcher_session_id IS NOT NULL
                  OR NEW.caller->>'kind' = 'session' THEN 'session'
                {agent_arm}
                WHEN NEW.parent_run_id IS NOT NULL THEN
                    (SELECT principal FROM wf_runs WHERE id = NEW.parent_run_id)
                ELSE 'operator'
            END;
            RETURN NEW;
        END
        $$
    """)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '5min'")
    _replace_function("WHEN NEW.as_agent_id IS NOT NULL THEN 'agent'")
    op.execute(r"""
        WITH RECURSIVE acts_for_agent(id) AS (
            SELECT id FROM wf_runs WHERE as_agent_id IS NOT NULL
            UNION
            SELECT child.id
              FROM wf_runs child
              JOIN acts_for_agent parent ON child.parent_run_id = parent.id
             WHERE child.principal = 'operator'
        )
        UPDATE wf_runs r
           SET principal = 'agent'
          FROM acts_for_agent a
         WHERE r.id = a.id
    """)


def downgrade() -> None:
    op.execute("UPDATE wf_runs SET principal = 'operator' WHERE principal = 'agent'")
    _replace_function("")
