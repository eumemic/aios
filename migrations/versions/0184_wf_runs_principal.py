"""Record who each workflow run acts for, immutably (#2467).

Revision ID: 0184
Revises: 0183

``wf_runs.principal`` is ``operator`` or ``session``. Authority checks read it
instead of ``launcher_session_id IS NULL``, which was unsound: that FK is
``ON DELETE SET NULL``, so deleting a session turned its live runs into operator
runs, and 0078 backfilled NULL for every agent-launched run that predated it.

A BEFORE INSERT trigger is the column's only writer. It stamps ``session`` when
a session launched or awaits the run, the parent's principal for a sub-run, and
``operator`` otherwise. Stamping in the database rather than in ``create_run``
covers every writer, including an application image from before this revision,
which inserts runs without the column during candidate admission and after an
application rollback. Nothing updates the column after insert.

Backfill. ``launcher_session_id`` can't classify existing rows (see above), so a
row is ``session`` when any surviving evidence says a session launched it or one
of its ancestors: a launcher, a session caller, or a trigger fire (every
existing trigger has an owner session; a new fire carries that owner as its
launcher, so the insert trigger doesn't read ``trigger_id``). A fire is
recognised by ``trigger_id`` or, for runs fired before 0182 added that column,
by the ``trigger_runs.result_id`` audit row, which has no FK to the owner session
and so survives its deletion. A parent edge never leads from a session run to an
operator run: sub-runs inherit their parent's principal, and the other parent
edges (a run's agent child calling a workflow, a trigger fire) always have a
launcher session. Everything else is ``operator``.

Mechanics. The column add, trigger and backfill run in one transaction under the
ADD COLUMN's ACCESS EXCLUSIVE lock, so an old writer blocked on that lock inserts
after COMMIT, when the trigger is installed. The constant default makes the add
catalog-only (no rewrite, no NOT NULL scan); the backfill rewrites only the
operator runs, with the CTE membership test in WHERE so it plans as a hash join;
then the default is dropped.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0184"
down_revision: str | None = "0183"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '5min'")
    op.execute("ALTER TABLE wf_runs ADD COLUMN principal text NOT NULL DEFAULT 'session'")
    op.execute(r"""
        CREATE FUNCTION _aios_stamp_wf_run_principal() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            NEW.principal := CASE
                WHEN NEW.launcher_session_id IS NOT NULL
                  OR NEW.caller->>'kind' = 'session' THEN 'session'
                WHEN NEW.parent_run_id IS NOT NULL THEN
                    (SELECT principal FROM wf_runs WHERE id = NEW.parent_run_id)
                ELSE 'operator'
            END;
            RETURN NEW;
        END
        $$
    """)
    op.execute(r"""
        CREATE TRIGGER wf_runs_stamp_principal_trg
        BEFORE INSERT ON wf_runs
        FOR EACH ROW
        EXECUTE FUNCTION _aios_stamp_wf_run_principal()
    """)
    op.execute(r"""
        WITH RECURSIVE acts_for_session(id) AS (
            SELECT id FROM wf_runs
             WHERE launcher_session_id IS NOT NULL
                OR caller->>'kind' = 'session'
                OR trigger_id IS NOT NULL
                OR id IN (SELECT result_id FROM trigger_runs WHERE result_id IS NOT NULL)
            UNION
            SELECT child.id
              FROM wf_runs child
              JOIN acts_for_session parent ON child.parent_run_id = parent.id
        )
        UPDATE wf_runs r
           SET principal = 'operator'
         WHERE NOT EXISTS (SELECT 1 FROM acts_for_session s WHERE s.id = r.id)
    """)
    op.execute("ALTER TABLE wf_runs ALTER COLUMN principal DROP DEFAULT")


def downgrade() -> None:
    op.execute("DROP TRIGGER wf_runs_stamp_principal_trg ON wf_runs")
    op.execute("DROP FUNCTION _aios_stamp_wf_run_principal()")
    op.execute("ALTER TABLE wf_runs DROP COLUMN principal")
