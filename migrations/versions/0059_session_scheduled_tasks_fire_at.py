"""Session scheduled tasks: one-shot via ``fire_at`` + NOTIFY trigger.

Two changes that together pivot ``session_scheduled_tasks`` from cron-only
to a unified scheduling substrate (cron + one-shot):

- ``schedule`` becomes nullable; new ``fire_at timestamptz`` column. CHECK
  constraint enforces exactly one of (``schedule``, ``fire_at``) is set —
  cron rows recur per ``schedule``; one-shot rows fire once at ``fire_at``
  and self-delete in the runner.
- AFTER INSERT/UPDATE/DELETE trigger emits ``pg_notify('aios_scheduled_tasks_due', id)``
  on changes that affect scheduling (``schedule``, ``fire_at``, ``enabled``,
  ``next_fire``). The trigger gates UPDATE notifies to those columns so
  scheduler-internal writes (``running_since`` cleared, ``last_fire_at``
  bumped, ``consecutive_failures`` advanced) don't wake the scheduler for
  no reason.

Revision ID: 0059
Revises: 0058
Create Date: 2026-05-23
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0059"
down_revision: str = "0058"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE session_scheduled_tasks ALTER COLUMN schedule DROP NOT NULL")
    op.execute("ALTER TABLE session_scheduled_tasks ADD COLUMN fire_at timestamptz")
    op.execute("""
        ALTER TABLE session_scheduled_tasks
        ADD CONSTRAINT sched_tasks_schedule_xor_fire_at
        CHECK (
            (schedule IS NOT NULL AND fire_at IS NULL)
            OR (schedule IS NULL AND fire_at IS NOT NULL)
        )
    """)

    # NOTIFY trigger: wakes the event-driven scheduler when something
    # scheduling-relevant changes. Gates UPDATE notifies to only the
    # columns that can affect the scheduler's MIN(next_event) query —
    # internal writes from the runner (clearing ``running_since`` on
    # completion, bumping ``last_fire_at`` and ``consecutive_failures``)
    # do NOT notify, which would otherwise cause the scheduler to wake
    # itself for its own bookkeeping writes.
    op.execute(r"""
        CREATE OR REPLACE FUNCTION notify_scheduled_tasks_due() RETURNS TRIGGER AS $$
        BEGIN
            IF (TG_OP = 'INSERT') THEN
                PERFORM pg_notify('aios_scheduled_tasks_due', NEW.id);
            ELSIF (TG_OP = 'UPDATE') THEN
                IF (
                    OLD.schedule IS DISTINCT FROM NEW.schedule
                    OR OLD.fire_at IS DISTINCT FROM NEW.fire_at
                    OR OLD.enabled IS DISTINCT FROM NEW.enabled
                    OR OLD.next_fire IS DISTINCT FROM NEW.next_fire
                ) THEN
                    PERFORM pg_notify('aios_scheduled_tasks_due', NEW.id);
                END IF;
            ELSIF (TG_OP = 'DELETE') THEN
                PERFORM pg_notify('aios_scheduled_tasks_due', OLD.id);
            END IF;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER session_scheduled_tasks_notify
        AFTER INSERT OR UPDATE OR DELETE ON session_scheduled_tasks
        FOR EACH ROW
        EXECUTE FUNCTION notify_scheduled_tasks_due()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS session_scheduled_tasks_notify ON session_scheduled_tasks")
    op.execute("DROP FUNCTION IF EXISTS notify_scheduled_tasks_due()")
    op.execute(
        "ALTER TABLE session_scheduled_tasks "
        "DROP CONSTRAINT IF EXISTS sched_tasks_schedule_xor_fire_at"
    )
    op.execute("ALTER TABLE session_scheduled_tasks DROP COLUMN IF EXISTS fire_at")
    # NB: can't restore schedule NOT NULL cleanly if any rows have NULL;
    # the downgrade simply leaves it nullable.
