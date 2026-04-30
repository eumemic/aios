"""Memory stores: workspace-scoped persistent text storage with audit trail.

Adds four tables for the memory store feature (parity with Anthropic Managed
Agents memory):

1. ``memory_stores`` — top-level resource with name/description/metadata and a
   gapless ``last_version_seq`` counter for the per-store version log.
2. ``memories`` — path-addressed text content within a store. Soft-deleted via
   ``deleted_at`` so version history retains the ``memory_id`` reference.
3. ``memory_versions`` — append-only, immutable audit trail. ``operation`` is
   one of ``created`` / ``modified`` / ``deleted``. Redaction nulls out the
   path/content/sha/size while preserving ``created_by`` + timestamps.
4. ``session_memory_stores`` — junction binding stores to sessions, with
   ``access`` (read_only|read_write) and ``instructions`` per attachment. Name
   and description are snapshotted at attach time so renames don't affect
   running sessions.

Revision ID: 0025
Revises: 0024
Create Date: 2026-04-29
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0025"
down_revision: str = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE memory_stores (
            id               text PRIMARY KEY,
            name             text NOT NULL,
            description      text NOT NULL DEFAULT '',
            metadata         jsonb NOT NULL DEFAULT '{}'::jsonb,
            last_version_seq bigint NOT NULL DEFAULT 0,
            created_at       timestamptz NOT NULL DEFAULT now(),
            updated_at       timestamptz NOT NULL DEFAULT now(),
            archived_at      timestamptz
        )
    """)

    op.execute(r"""
        CREATE TABLE memories (
            id                  text PRIMARY KEY,
            memory_store_id     text NOT NULL REFERENCES memory_stores(id) ON DELETE CASCADE,
            path                text NOT NULL,
            content             text NOT NULL,
            content_sha256      text NOT NULL,
            content_size_bytes  integer NOT NULL,
            current_version_id  text,
            created_at          timestamptz NOT NULL DEFAULT now(),
            updated_at          timestamptz NOT NULL DEFAULT now(),
            deleted_at          timestamptz,
            CHECK (path ~ '^(/[^/\x00]+)+$'),
            CHECK (content_size_bytes <= 102400),
            CHECK (content_size_bytes = octet_length(content))
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX memories_store_path_active
            ON memories (memory_store_id, path)
            WHERE deleted_at IS NULL
    """)

    op.execute("""
        CREATE TABLE memory_versions (
            id                  text PRIMARY KEY,
            memory_store_id     text NOT NULL REFERENCES memory_stores(id) ON DELETE CASCADE,
            memory_id           text NOT NULL,
            seq                 bigint NOT NULL,
            operation           text NOT NULL
                CHECK (operation IN ('created', 'modified', 'deleted')),
            path                text,
            content             text,
            content_sha256      text,
            content_size_bytes  integer,
            created_by_type     text NOT NULL
                CHECK (created_by_type IN ('api_actor', 'session_actor')),
            created_by_ref      text NOT NULL,
            created_at          timestamptz NOT NULL DEFAULT now(),
            redacted_at         timestamptz,
            redacted_by_type    text
                CHECK (redacted_by_type IS NULL
                       OR redacted_by_type IN ('api_actor', 'session_actor')),
            redacted_by_ref     text,
            UNIQUE (memory_store_id, seq)
        )
    """)
    op.execute("""
        CREATE INDEX memory_versions_by_memory
            ON memory_versions (memory_id, created_at DESC)
    """)

    op.execute("""
        CREATE TABLE session_memory_stores (
            session_id              text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            memory_store_id         text NOT NULL REFERENCES memory_stores(id),
            rank                    integer NOT NULL,
            access                  text NOT NULL
                CHECK (access IN ('read_only', 'read_write')),
            instructions            text NOT NULL DEFAULT '',
            name_at_attach          text NOT NULL,
            description_at_attach   text NOT NULL,
            PRIMARY KEY (session_id, memory_store_id),
            CHECK (rank BETWEEN 0 AND 7),
            CHECK (length(instructions) <= 4096)
        )
    """)
    op.execute("""
        CREATE INDEX session_memory_stores_by_session
            ON session_memory_stores (session_id, rank)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS session_memory_stores")
    op.execute("DROP TABLE IF EXISTS memory_versions")
    op.execute("DROP TABLE IF EXISTS memories")
    op.execute("DROP TABLE IF EXISTS memory_stores")
