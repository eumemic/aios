"""Workflows: durable runtime core (Block 1).

A *workflow* is a deterministic-Python orchestrator — the dual of an agent.
This migration lays the durable substrate for the v1 runtime core:

- ``workflows``      — immutable, versioned script definitions.
- ``wf_runs``        — durable execution instances. Each run snapshots the
  in-force ``script`` (and its ``script_sha``) at creation, so every wake
  execs that exact immutable source regardless of later definition edits.
  ``status`` is *persisted* for runs (unlike sessions, whose status is now
  derived — #732); the run loop writes ``suspended``/``completed``/``errored``.
- ``wf_run_events``  — each run's append-only journal (the replay-with-memo
  source). ``UNIQUE(run_id, call_key, type)`` is the memo; ``UNIQUE(run_id,
  seq)`` enforces the gapless sequence.
- ``wf_run_signals`` — a side-marker table. External resumes (gate) and, later,
  child completions write *here* (idempotent on the composite PK) and defer a
  run wake; the durable ``call_result`` is journaled exclusively by the next
  ``run_workflow_step`` harvest, so ``wf_run_events`` keeps a single writer.

Additive, nullable columns ``sessions.parent_run_id`` + ``sessions.origin``
pre-shape the session core for Block 2 (workflow-spawned ``agent()`` children)
and the Block 3 priority lane; they are unused in Block 1.

All four tables are brand-new (empty at creation), so plain ``CREATE INDEX``
inside the migration transaction is safe — no ``CONCURRENTLY`` needed (cf.
0062, which indexed the live ``events`` table).

Revision ID: 0064
Revises: 0063
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0064"
down_revision: str = "0063"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE workflows (
            id            text PRIMARY KEY,
            account_id    text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            name          text NOT NULL,
            version       integer NOT NULL DEFAULT 1,
            script        text NOT NULL,
            input_schema  jsonb,
            output_schema jsonb,
            created_at    timestamptz NOT NULL DEFAULT now(),
            updated_at    timestamptz NOT NULL DEFAULT now(),
            UNIQUE (account_id, name, version)
        )
    """)

    op.execute(r"""
        CREATE TABLE wf_runs (
            id             text PRIMARY KEY,
            workflow_id    text NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
            account_id     text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            parent_run_id  text REFERENCES wf_runs(id) ON DELETE SET NULL,
            script         text NOT NULL,
            script_sha     text NOT NULL,
            status         text NOT NULL DEFAULT 'pending'
                           CHECK (status IN ('pending','running','suspended','completed','errored')),
            input          jsonb,
            output         jsonb,
            last_event_seq bigint NOT NULL DEFAULT 0,
            created_at     timestamptz NOT NULL DEFAULT now(),
            updated_at     timestamptz NOT NULL DEFAULT now(),
            archived_at    timestamptz
        )
    """)
    # Drives the startup + 30s periodic re-enqueue sweep (non-terminal, live runs).
    op.execute(r"""
        CREATE INDEX wf_runs_active_idx ON wf_runs (status)
            WHERE archived_at IS NULL AND status IN ('pending','running','suspended')
    """)
    op.execute("CREATE INDEX wf_runs_workflow_idx ON wf_runs (workflow_id, created_at DESC)")

    op.execute(r"""
        CREATE TABLE wf_run_events (
            id          text PRIMARY KEY,
            run_id      text NOT NULL REFERENCES wf_runs(id) ON DELETE CASCADE,
            seq         bigint NOT NULL,
            type        text NOT NULL
                        CHECK (type IN ('run_started','call_started','call_result','run_completed')),
            call_key    text,
            payload     jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at  timestamptz NOT NULL DEFAULT now(),
            UNIQUE (run_id, seq),
            UNIQUE (run_id, call_key, type)
        )
    """)
    op.execute("CREATE INDEX wf_run_events_run_idx ON wf_run_events (run_id, seq)")

    op.execute(r"""
        CREATE TABLE wf_run_signals (
            run_id       text NOT NULL REFERENCES wf_runs(id) ON DELETE CASCADE,
            call_key     text NOT NULL,
            kind         text NOT NULL CHECK (kind IN ('gate_resume','child_done')),
            result       jsonb,
            delivered_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (run_id, call_key)
        )
    """)

    # Additive, nullable pre-shape on the session core (unused in Block 1).
    op.execute(
        "ALTER TABLE sessions ADD COLUMN parent_run_id text REFERENCES wf_runs(id) ON DELETE SET NULL"
    )
    op.execute(
        "ALTER TABLE sessions ADD COLUMN origin text NOT NULL DEFAULT 'foreground' "
        "CHECK (origin IN ('foreground','background'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS origin")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS parent_run_id")
    op.execute("DROP TABLE IF EXISTS wf_run_signals")
    op.execute("DROP TABLE IF EXISTS wf_run_events")
    op.execute("DROP TABLE IF EXISTS wf_runs")
    op.execute("DROP TABLE IF EXISTS workflows")
