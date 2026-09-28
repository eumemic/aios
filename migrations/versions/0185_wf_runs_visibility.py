"""Record who may read each workflow run through agent tools (#2468).

Revision ID: 0185
Revises: 0184

``wf_runs.visibility`` is ``account`` (any session in the account, the old
behavior) or ``session`` (only the run's launching session). The operator API
ignores it. A model-dispatch run, launched by a workflow-as-model park, is
``session``: its input is the bound session's full request, and the agent
run-read tools would otherwise show that conversation to every session in the
account. Sub-runs inherit their parent's visibility.

As with ``principal`` (0184), a BEFORE INSERT trigger is the column's only
writer, so an application image from before this revision stamps its runs
correctly too. A model-dispatch run is recognised by the ``purpose`` its park
writes on the caller edge.

Mechanics follow 0184: a catalog-only constant default, a backfill that
rewrites only the model-dispatch subtrees (membership in WHERE, so it plans as a
hash join), then the default is dropped.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0185"
down_revision: str | None = "0184"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '5min'")
    op.execute("ALTER TABLE wf_runs ADD COLUMN visibility text NOT NULL DEFAULT 'account'")
    op.execute(r"""
        CREATE FUNCTION _aios_stamp_wf_run_visibility() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            NEW.visibility := CASE
                WHEN NEW.caller->>'purpose' = 'model_dispatch' THEN 'session'
                WHEN NEW.parent_run_id IS NOT NULL THEN
                    (SELECT visibility FROM wf_runs WHERE id = NEW.parent_run_id)
                ELSE 'account'
            END;
            RETURN NEW;
        END
        $$
    """)
    op.execute(r"""
        CREATE TRIGGER wf_runs_stamp_visibility_trg
        BEFORE INSERT ON wf_runs
        FOR EACH ROW
        EXECUTE FUNCTION _aios_stamp_wf_run_visibility()
    """)
    op.execute(r"""
        WITH RECURSIVE session_private(id) AS (
            SELECT id FROM wf_runs WHERE caller->>'purpose' = 'model_dispatch'
            UNION
            SELECT child.id
              FROM wf_runs child
              JOIN session_private parent ON child.parent_run_id = parent.id
        )
        UPDATE wf_runs r
           SET visibility = 'session'
         WHERE EXISTS (SELECT 1 FROM session_private s WHERE s.id = r.id)
    """)
    op.execute("ALTER TABLE wf_runs ALTER COLUMN visibility DROP DEFAULT")


def downgrade() -> None:
    op.execute("DROP TRIGGER wf_runs_stamp_visibility_trg ON wf_runs")
    op.execute("DROP FUNCTION _aios_stamp_wf_run_visibility()")
    op.execute("ALTER TABLE wf_runs DROP COLUMN visibility")
