"""Session scheduled tasks (#636).

Each row is a per-session cron entry: a schedule + bash command + metadata
plus runtime fields (next_fire, running_since, last_fire_*, consecutive_failures).
The session-scheduler-tick selects due rows under SKIP LOCKED, sets
``running_since`` (to prevent overlap) and advances ``next_fire`` in the
same transaction, then enqueues a procrastinate fire-job per claimed row.

The partial index ``sched_tasks_due`` is the scheduler-tick's hot path.

Revision ID: 0058
Revises: 0057
Create Date: 2026-05-22
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0058"
down_revision: str = "0057"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE session_scheduled_tasks (
            id                    text PRIMARY KEY,
            session_id            text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            account_id            text NOT NULL,
            name                  text NOT NULL,
            schedule              text NOT NULL,
            command               text NOT NULL,
            enabled               boolean NOT NULL DEFAULT true,
            timeout_seconds       integer NOT NULL DEFAULT 300,
            max_output_bytes      integer NOT NULL DEFAULT 65536,
            next_fire             timestamptz,
            running_since         timestamptz,
            last_fire_at          timestamptz,
            last_fire_status      text
                CHECK (last_fire_status IN ('ok', 'error', 'timeout', 'skipped')),
            consecutive_failures  integer NOT NULL DEFAULT 0,
            metadata              jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at            timestamptz NOT NULL DEFAULT now(),
            updated_at            timestamptz NOT NULL DEFAULT now(),
            UNIQUE (session_id, name),
            CHECK (name ~ '^[a-zA-Z0-9][a-zA-Z0-9_-]*$')
        )
    """)
    op.execute("""
        CREATE INDEX sched_tasks_by_session
            ON session_scheduled_tasks (session_id)
    """)
    # Partial index for the scheduler-tick hot path. Covers BOTH the
    # idle branch (``running_since IS NULL``) and the stale-recovery
    # branch (``running_since <= cutoff``) of the SELECT in
    # ``fetch_and_claim_due_scheduled_tasks`` — the predicate is kept
    # broad enough that stuck rows don't fall out of the index and
    # force a table scan during recovery.
    op.execute("""
        CREATE INDEX sched_tasks_due
            ON session_scheduled_tasks (next_fire)
            WHERE enabled AND next_fire IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS session_scheduled_tasks")
