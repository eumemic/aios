"""Initial schema for aios v1 phase 1.

This migration creates the five core tables:

* ``credentials`` — encrypted API keys (libsodium ciphertext + nonce)
* ``environments`` — sandbox environment definitions (name only in v1)
* ``agents`` — agent records (Phase 1: no version table yet; the
  ``agent_versions`` table is introduced in a later migration when we add
  Anthropic-style immutable versioning in Phase 4)
* ``sessions`` — running session state, including the workspace volume path,
  the optional Docker container assignment, and the harness lease columns
* ``events`` — append-only session event log; gapless ``seq`` per session
  guaranteed by the ``last_event_seq`` row counter on the ``sessions`` table

Revision ID: 0001
Revises:
Create Date: 2026-04-10
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE credentials (
            id           text PRIMARY KEY,
            name         text NOT NULL,
            provider     text NOT NULL,
            ciphertext   bytea NOT NULL,
            nonce        bytea NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            updated_at   timestamptz NOT NULL DEFAULT now(),
            archived_at  timestamptz
        );
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX credentials_name_uniq "
        "ON credentials (name) WHERE archived_at IS NULL;"
    )

    op.execute(
        """
        CREATE TABLE environments (
            id           text PRIMARY KEY,
            name         text NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            archived_at  timestamptz
        );
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX environments_name_uniq "
        "ON environments (name) WHERE archived_at IS NULL;"
    )

    # Phase 1 agents table: no agent_versions table yet. The agent record
    # carries model/system/tools directly. Phase 4 migrates this to an
    # immutable agent_versions table with backfill.
    op.execute(
        """
        CREATE TABLE agents (
            id              text PRIMARY KEY,
            name            text NOT NULL,
            model           text NOT NULL,
            system          text NOT NULL DEFAULT '',
            tools           jsonb NOT NULL DEFAULT '[]'::jsonb,
            credential_id   text REFERENCES credentials(id),
            description     text,
            metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
            window_min      integer NOT NULL DEFAULT 50000,
            window_max      integer NOT NULL DEFAULT 150000,
            created_at      timestamptz NOT NULL DEFAULT now(),
            updated_at      timestamptz NOT NULL DEFAULT now(),
            archived_at     timestamptz
        );
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX agents_name_uniq "
        "ON agents (name) WHERE archived_at IS NULL;"
    )

    op.execute(
        """
        CREATE TABLE sessions (
            id                      text PRIMARY KEY,
            agent_id                text NOT NULL REFERENCES agents(id),
            environment_id          text NOT NULL REFERENCES environments(id),
            title                   text,
            metadata                jsonb NOT NULL DEFAULT '{}'::jsonb,
            status                  text NOT NULL
                CHECK (status IN ('running','idle','terminated')),
            stop_reason             text,
            workspace_volume_path   text NOT NULL,
            container_id            text,
            lease_worker_id         text,
            lease_expires_at        timestamptz,
            last_event_seq          bigint NOT NULL DEFAULT 0,
            created_at              timestamptz NOT NULL DEFAULT now(),
            updated_at              timestamptz NOT NULL DEFAULT now(),
            archived_at             timestamptz
        );
        """
    )
    op.execute(
        "CREATE INDEX sessions_agent_idx "
        "ON sessions (agent_id, created_at DESC);"
    )
    op.execute(
        "CREATE INDEX sessions_status_idx "
        "ON sessions (status) WHERE archived_at IS NULL;"
    )
    op.execute(
        "CREATE INDEX sessions_lease_idx "
        "ON sessions (lease_expires_at) WHERE lease_worker_id IS NOT NULL;"
    )

    op.execute(
        """
        CREATE TABLE events (
            id          text PRIMARY KEY,
            session_id  text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            seq         bigint NOT NULL,
            kind        text NOT NULL
                CHECK (kind IN ('message','lifecycle','span','interrupt')),
            data        jsonb NOT NULL,
            created_at  timestamptz NOT NULL DEFAULT now(),
            UNIQUE (session_id, seq)
        );
        """
    )
    op.execute("CREATE INDEX events_session_seq_idx ON events (session_id, seq);")
    op.execute("CREATE INDEX events_created_brin ON events USING BRIN (created_at);")
    op.execute(
        "CREATE INDEX events_session_message_seq_idx "
        "ON events (session_id, seq) WHERE kind = 'message';"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS events;")
    op.execute("DROP TABLE IF EXISTS sessions;")
    op.execute("DROP TABLE IF EXISTS agents;")
    op.execute("DROP TABLE IF EXISTS environments;")
    op.execute("DROP TABLE IF EXISTS credentials;")
