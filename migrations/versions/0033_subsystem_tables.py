"""Connector subsystem tables alongside old, with backfill (#328 PR 2/8).

Lands the seven ``aios_connectors`` subsystem tables (``connectors``,
``bindings``, ``chat_sessions``, ``routing_rules``, ``runtimes``,
``runtime_tokens``, ``inbound_acks``) alongside today's connector tables.
Old tables stay authoritative this PR; PR 4 is the code switch. Backfill
seeds the new tables from existing data where the mapping is mechanical
(curation, chat ledger, dedup). Runtime liveness and bearer tokens stay
empty — PR 5/6 populate them.

Every new table reserves ``user_id text`` for the future multi-tenant
migration (no enforcement here; PR 3 adds it to existing core tables).

Revision ID: 0033
Revises: 0032
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0033"
down_revision: str = "0032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---- connectors: connector-type catalog -------------------------------
    # Natural-key PK: the connector type IS the identity (a small fixed set
    # — "signal", "telegram", "echo-http", …). Future sharding across
    # multiple runtimes of the same type still keys on the type, so a
    # surrogate id would just add indirection.
    op.execute(
        """
        CREATE TABLE connectors (
            connector     text PRIMARY KEY,
            tools_schema  jsonb NOT NULL DEFAULT '[]'::jsonb,
            user_id       text,
            created_at    timestamptz NOT NULL DEFAULT now(),
            updated_at    timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    # Most recently updated active connection's tools jsonb wins per
    # connector type; operator reconciliation post-PR 5 supersedes.
    op.execute(
        """
        INSERT INTO connectors (connector, tools_schema, created_at, updated_at)
        SELECT DISTINCT ON (connector)
               connector,
               COALESCE(tools, '[]'::jsonb),
               now(),
               now()
          FROM connections
         WHERE archived_at IS NULL
         ORDER BY connector, updated_at DESC
        """
    )

    # ---- bindings: unit of curation ---------------------------------------
    op.execute(
        """
        CREATE TABLE bindings (
            id                   text PRIMARY KEY,
            connection_id        text NOT NULL REFERENCES connections(id),
            mode                 text NOT NULL,
            session_id           text REFERENCES sessions(id),
            session_template_id  text REFERENCES session_templates(id),
            user_id              text,
            created_at           timestamptz NOT NULL DEFAULT now(),
            archived_at          timestamptz,
            CONSTRAINT bindings_mode_ck CHECK (mode IN ('single_session', 'per_chat')),
            CONSTRAINT bindings_target_matches_mode_ck CHECK (
                (mode = 'single_session'
                    AND session_id IS NOT NULL
                    AND session_template_id IS NULL)
             OR (mode = 'per_chat'
                    AND session_template_id IS NOT NULL
                    AND session_id IS NULL)
            )
        )
        """
    )
    # At most one active binding per connection (matches today's invariant
    # that a connection has at most one curation target).
    op.execute(
        "CREATE UNIQUE INDEX bindings_connection_active_uniq "
        "ON bindings (connection_id) WHERE archived_at IS NULL"
    )
    op.execute(
        "CREATE INDEX bindings_session_id_idx "
        "ON bindings (session_id) WHERE archived_at IS NULL"
    )
    op.execute(
        "CREATE INDEX bindings_session_template_id_idx "
        "ON bindings (session_template_id) WHERE archived_at IS NULL"
    )

    # ``gen_random_uuid()`` is built into PG 13+; the ``bnd_`` prefix
    # matches ``aios.ids.BINDING`` for log readability. The body is hex
    # rather than a Crockford-ULID — pure-SQL ULID generation isn't
    # worth the complexity here, and no current code path runs
    # ``split_id`` on a binding id. See note on ``BINDING`` in ids.py.
    op.execute(
        """
        INSERT INTO bindings (id, connection_id, mode, session_id,
                              session_template_id, created_at)
        SELECT 'bnd_' || REPLACE(gen_random_uuid()::text, '-', ''),
               id,
               CASE WHEN session_id IS NOT NULL THEN 'single_session'
                    ELSE 'per_chat' END,
               session_id,
               session_template_id,
               COALESCE(attached_at, created_at)
          FROM connections
         WHERE archived_at IS NULL
           AND (session_id IS NOT NULL OR session_template_id IS NOT NULL)
        """
    )

    # ---- chat_sessions: per_chat ledger ----------------------------------
    op.execute(
        """
        CREATE TABLE chat_sessions (
            connection_id  text NOT NULL REFERENCES connections(id),
            chat_id        text NOT NULL,
            session_id     text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            user_id        text,
            created_at     timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (connection_id, chat_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX chat_sessions_session_idx ON chat_sessions (session_id)"
    )
    op.execute(
        """
        INSERT INTO chat_sessions (connection_id, chat_id, session_id, created_at)
        SELECT connection_id, chat_id, session_id, created_at
          FROM connection_chat_sessions
        """
    )

    # ---- routing_rules: per-binding prefix demux (DDL only this PR) ------
    # ``target_id`` is plain text rather than a polymorphic FK — SQL can't
    # FK to a union of (sessions, session_templates) and the application
    # layer resolves it by ``target_type``. Resolver lands in PR 4.
    op.execute(
        """
        CREATE TABLE routing_rules (
            id           text PRIMARY KEY,
            binding_id   text NOT NULL REFERENCES bindings(id) ON DELETE CASCADE,
            prefix       text NOT NULL,
            target_type  text NOT NULL,
            target_id    text NOT NULL,
            user_id      text,
            created_at   timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT routing_rules_target_type_ck
                CHECK (target_type IN ('session', 'session_template'))
        )
        """
    )
    op.execute(
        "CREATE INDEX routing_rules_binding_id_idx ON routing_rules (binding_id)"
    )

    # ---- runtimes: container liveness ------------------------------------
    op.execute(
        """
        CREATE TABLE runtimes (
            id                 text PRIMARY KEY,
            connector          text NOT NULL REFERENCES connectors(connector),
            user_id            text,
            created_at         timestamptz NOT NULL DEFAULT now(),
            last_heartbeat_at  timestamptz
        )
        """
    )
    op.execute(
        "CREATE INDEX runtimes_connector_idx ON runtimes (connector)"
    )

    # ---- runtime_tokens: per-connector-type bearer auth ------------------
    # No ``connection_id`` FK: one bearer scopes N connections of one type,
    # discovered at runtime via SSE add/remove (PR 5). SHA-256 hash with
    # soft-revocation, same shape as ``connector_tokens`` (which it replaces).
    op.execute(
        """
        CREATE TABLE runtime_tokens (
            id            text PRIMARY KEY,
            connector     text NOT NULL REFERENCES connectors(connector),
            token_hash    text NOT NULL UNIQUE,
            label         text,
            user_id       text,
            created_at    timestamptz NOT NULL DEFAULT now(),
            last_used_at  timestamptz,
            revoked_at    timestamptz
        )
        """
    )
    op.execute(
        "CREATE INDEX runtime_tokens_connector_idx "
        "ON runtime_tokens (connector) WHERE revoked_at IS NULL"
    )

    # ---- inbound_acks: dedup ledger (rename target) ----------------------
    op.execute(
        """
        CREATE TABLE inbound_acks (
            connector      text NOT NULL,
            account        text NOT NULL,
            event_id       text NOT NULL,
            appended_seq   bigint NOT NULL,
            user_id        text,
            appended_at    timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (connector, account, event_id)
        )
        """
    )
    op.execute(
        """
        INSERT INTO inbound_acks (connector, account, event_id,
                                  appended_seq, appended_at)
        SELECT connector, account, event_id, appended_seq, appended_at
          FROM connector_inbound_acks
        """
    )


def downgrade() -> None:
    # Data-lossy: dropping the new tables doesn't reconstitute prior state
    # if the new tables were ever written to as authoritative. Provided so
    # ``alembic downgrade`` doesn't error, not as a real rollback path —
    # see migrations/versions/0027_connector_redesign.py for the same posture.
    op.execute("DROP TABLE IF EXISTS inbound_acks")
    op.execute("DROP TABLE IF EXISTS runtime_tokens")
    op.execute("DROP TABLE IF EXISTS runtimes")
    op.execute("DROP TABLE IF EXISTS routing_rules")
    op.execute("DROP TABLE IF EXISTS chat_sessions")
    op.execute("DROP TABLE IF EXISTS bindings")
    op.execute("DROP TABLE IF EXISTS connectors")
