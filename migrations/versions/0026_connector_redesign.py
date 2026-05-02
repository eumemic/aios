"""Connector redesign: collapse three routing tables into one.

Replaces ``connections`` + ``channel_bindings`` + ``connection_routing_rules``
with a single ``connections`` table whose unique
``(connector, account) WHERE archived_at IS NULL`` enforces "one session per
account" by schema, plus two routing modes:

* ``single_session`` — ``session_id`` populated; one attached session.
* ``per_chat`` — ``session_template_id`` populated; sessions auto-spawn per
  chat partner via ``connection_chat_sessions``.

Adds ``session_templates`` (frozen recipe for per_chat spawn),
``connection_chat_sessions`` (chat → session map, PK ``(connection_id,
chat_id)``), ``connector_inbound_acks`` (dedup ledger written in the same
txn as ``append_event`` — at-most-once event append), and
``sessions.spawned_from_connection_id`` (per_chat origin pointer, also the
outbound-permission grant for that session).

Pre-1.0: dev/prod rows in the deleted tables are dropped with no migration
path.  See plan §"PR1: Schema cutover" + resolved decision #1 (no
``vault_credentials.account_id`` to drop — that column never existed).

Revision ID: 0026
Revises: 0025
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0026"
down_revision: str = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop old routing tables in dependency order.  ``connection_id`` FKs
    # on bindings/rules cascade-delete when the parent connection is dropped,
    # but we drop them explicitly first to keep the order obvious.
    op.execute("DROP TABLE IF EXISTS routing_rules")
    op.execute("DROP TABLE IF EXISTS channel_bindings")
    op.execute("DROP TABLE IF EXISTS connections")

    # Session templates first — connections reference it.
    op.execute("""
        CREATE TABLE session_templates (
            id                text PRIMARY KEY,
            name              text NOT NULL,
            agent_id          text NOT NULL REFERENCES agents(id),
            agent_version     integer,
            environment_id    text NOT NULL REFERENCES environments(id),
            vault_ids         text[] NOT NULL DEFAULT '{}',
            memory_store_ids  text[] NOT NULL DEFAULT '{}',
            metadata          jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at        timestamptz NOT NULL DEFAULT now(),
            updated_at        timestamptz NOT NULL DEFAULT now(),
            archived_at       timestamptz
        )
    """)
    op.execute(
        "CREATE UNIQUE INDEX session_templates_name_uniq "
        "ON session_templates (name) WHERE archived_at IS NULL"
    )

    # New connections table.  Three valid shapes:
    #   detached    — both session_id and session_template_id NULL
    #   single_session — session_id NOT NULL, session_template_id NULL
    #   per_chat    — session_id NULL, session_template_id NOT NULL
    op.execute("""
        CREATE TABLE connections (
            id                   text PRIMARY KEY,
            connector            text NOT NULL,
            account              text NOT NULL,
            session_id           text REFERENCES sessions(id),
            session_template_id  text REFERENCES session_templates(id),
            metadata             jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at           timestamptz NOT NULL DEFAULT now(),
            attached_at          timestamptz,
            updated_at           timestamptz NOT NULL DEFAULT now(),
            archived_at          timestamptz,
            CONSTRAINT connections_one_mode_ck CHECK (
                (session_id IS NULL AND session_template_id IS NULL)
                OR (session_id IS NOT NULL AND session_template_id IS NULL)
                OR (session_id IS NULL AND session_template_id IS NOT NULL)
            )
        )
    """)
    op.execute(
        "CREATE UNIQUE INDEX connections_active_account_uniq "
        "ON connections (connector, account) WHERE archived_at IS NULL"
    )

    # Per-chat session ledger.  PK ``(connection_id, chat_id)`` doubles as
    # the race-safe insert target — the inbound handler does
    # ``INSERT ... ON CONFLICT DO NOTHING RETURNING session_id`` and on
    # empty RETURNING re-reads the existing row.
    op.execute("""
        CREATE TABLE connection_chat_sessions (
            connection_id  text NOT NULL REFERENCES connections(id) ON DELETE CASCADE,
            chat_id        text NOT NULL,
            session_id     text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            created_at     timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (connection_id, chat_id)
        )
    """)
    op.execute(
        "CREATE INDEX connection_chat_sessions_session_idx "
        "ON connection_chat_sessions (session_id)"
    )

    # Origin pointer for per_chat-spawned sessions.  Doubles as the
    # outbound-permission grant: tools that take an ``account`` argument
    # can validate against either a connection attached to the calling
    # session OR the connection that spawned the session.
    op.execute(
        "ALTER TABLE sessions ADD COLUMN spawned_from_connection_id text "
        "REFERENCES connections(id)"
    )

    # Inbound dedup ledger.  Written in the same txn as ``append_event`` —
    # ``ON CONFLICT DO NOTHING`` is the dedup mechanism, giving at-most-once
    # event append.  Spool-side ack still runs after commit (different
    # purpose: spool pruning, not dedup).  See PR3 for the inbound handler;
    # the table ships in PR1 so the schema is final before PR2 builds
    # against it.
    op.execute("""
        CREATE TABLE connector_inbound_acks (
            connector      text NOT NULL,
            account        text NOT NULL,
            event_id       text NOT NULL,
            appended_seq   bigint NOT NULL,
            appended_at    timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (connector, account, event_id)
        )
    """)


def downgrade() -> None:
    # Data-lossy: dropping the new tables doesn't reconstitute the old
    # routing schema, and pre-1.0 rows in the old tables were dropped on
    # upgrade.  Provided so alembic ``downgrade`` doesn't error, not as a
    # rollback path.
    op.execute("DROP TABLE IF EXISTS connector_inbound_acks")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS spawned_from_connection_id")
    op.execute("DROP TABLE IF EXISTS connection_chat_sessions")
    op.execute("DROP TABLE IF EXISTS connections")
    op.execute("DROP TABLE IF EXISTS session_templates")
