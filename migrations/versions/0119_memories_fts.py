"""Full-text search over ``memories.content`` for the ``memory_search`` builtin.

This is the first FTS in the codebase. It is purely additive:

1. A generated STORED ``content_tsv`` column on ``memories`` —
   ``to_tsvector('english', content)``. Generated STORED means Postgres
   maintains the vector itself: it can never drift from ``content`` (no
   trigger, no ``append_event`` coupling, no hand-maintained scalar).
   ``memories.content`` is ``text NOT NULL`` with a 100 KiB CHECK
   (``0025_memory_stores.py``), so the generated column is bounded.
   Locale is hardcoded to ``'english'`` for v1 — there is no per-store
   locale concept today; generalize later.
2. A GIN index ``memories_content_tsv_gin`` on ``content_tsv`` — the
   point of the slice (ranked, stemmed full-text recall instead of a
   linear ILIKE / ``rg`` over the network-backed mounts).
3. A session-scoped ``memories_search`` VIEW, mirroring ``events_search``
   (``0010_events_search_view.py``): scoped via
   ``current_setting('app.session_id', true)`` joined through
   ``session_memory_stores``, so a session only ever sees memories of its
   own attached stores (the cross-session / cross-tenant isolation
   invariant). Soft-deleted memories (``deleted_at IS NOT NULL``) are
   excluded.

Additive ``ADD COLUMN ... GENERATED`` + ``CREATE INDEX`` + ``CREATE VIEW``
deploy cleanly in the new-code/old-schema window — no expand/contract.

Revision ID: 0119
Revises: 0118
Create Date: 2026-06-23
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0119"
down_revision: str = "0118"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE memories
            ADD COLUMN content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    """)
    op.execute("""
        CREATE INDEX memories_content_tsv_gin
            ON memories USING gin (content_tsv)
    """)
    op.execute("""
        CREATE VIEW memories_search AS
        SELECT
            m.id,
            sm.memory_store_id,
            sm.name_at_attach AS store,
            m.path,
            m.content,
            m.content_size_bytes,
            m.updated_at,
            m.content_tsv
        FROM memories m
        JOIN session_memory_stores sm
          ON sm.memory_store_id = m.memory_store_id
         AND sm.session_id = current_setting('app.session_id', true)
        WHERE m.deleted_at IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS memories_search")
    op.execute("DROP INDEX IF EXISTS memories_content_tsv_gin")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS content_tsv")
