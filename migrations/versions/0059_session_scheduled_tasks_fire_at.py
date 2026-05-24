"""Session scheduled tasks: one-shot via ``fire_at`` + NOTIFY trigger.

Two changes that together pivot ``session_scheduled_tasks`` from cron-only
to a unified scheduling substrate (cron + one-shot):

- ``schedule`` becomes nullable; new ``fire_at timestamptz`` column. CHECK
  constraint enforces exactly one of (``schedule``, ``fire_at``) is set —
  cron rows recur per ``schedule``; one-shot rows fire once at ``fire_at``
  and self-delete in the runner.
- AFTER INSERT/UPDATE/DELETE trigger emits ``pg_notify('aios_scheduled_tasks_due', id)``
  on changes that can shift the next-due time the scheduler is waiting for:
  ``schedule``, ``fire_at``, ``enabled``, AND the runner-clear edge
  (``OLD.running_since IS NOT NULL AND NEW.running_since IS NULL``).
  The runner-clear edge is load-bearing for short-period cron rows: after
  the post-claim recompute, the scheduler sees the row contributing
  ``GREATEST(next_fire, running_since + stale_threshold)`` ≈ now + 2h,
  clamps sleep to the 1h heartbeat, and would otherwise miss the rapid
  cadence of a sub-hourly cron entirely. The runner's
  ``record_scheduled_task_fire`` clears ``running_since``; that NOTIFY
  is what lets the scheduler re-compute MIN against the now-eligible
  ``next_fire``. Other scheduler-internal writes
  (``last_fire_at`` / ``consecutive_failures`` / ``updated_at``) stay
  outside the gate.

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

    # NOTIFY trigger: wakes the event-driven scheduler when something it
    # can't predict from its own next-MIN cache changes. Gates UPDATE
    # notifies to (1) user-facing fields the scheduler must react to
    # (``schedule``, ``fire_at``, ``enabled``) and (2) the runner-clear
    # edge — when ``running_since`` transitions from NOT NULL to NULL,
    # the row goes from "in-flight (contributes running_since + stale to
    # MIN)" back to "eligible (contributes next_fire to MIN)", which the
    # scheduler must see to honor short-period cron cadences. Without the
    # runner-clear NOTIFY, a 1-minute cron would only fire once per hour
    # because the scheduler sleeps until heartbeat after seeing the
    # in-flight contribution. Other scheduler-internal writes
    # (``last_fire_at`` / ``consecutive_failures`` / ``updated_at`` /
    # ``next_fire`` advanced post-claim for cron rows) stay outside the
    # gate — they don't change which row is next-due.
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
                    OR (OLD.running_since IS NOT NULL AND NEW.running_since IS NULL)
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
