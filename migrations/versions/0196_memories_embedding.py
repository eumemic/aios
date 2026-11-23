"""pgvector embedding on ``memories`` for hybrid semantic+keyword recall (#1629).

Extends the FTS slice (0119) with a second, additive ranking signal:

1. ``CREATE EXTENSION IF NOT EXISTS vector`` — the first extension in the tree.
   Requires a pgvector-bearing Postgres (CI/test image: ``pgvector/pgvector:pg16``).
2. ``memories.embedding vector(1536)`` — NULLable; absence is a row property
   (not-yet-embedded / provider ``Skipped``), excluded from the vector arm.
   The dimension is immutable per column: a model change is a re-embed
   migration, never a runtime switch.
3. ``memories.embedding_model text`` — the model that produced ``embedding``.
4. HNSW cosine index ``memories_embedding_hnsw`` (no training step, unlike IVFFlat).
5. ``memories_search`` view recreated to also expose ``embedding`` — same
   per-session scoping as 0119, so both ranking arms inherit isolation.

Unlike ``content_tsv`` (GENERATED STORED) the embedding needs an async model
call, so it is written explicitly at the service layer (post-commit,
DECOUPLED). Upgrade is purely additive; ``DROP COLUMN`` lives only in downgrade.

Revision ID: 0196
Revises: 0195
Create Date: 2026-11-22
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0196"
down_revision: str | None = "0195"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_VIEW_0119 = """
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
"""


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("ALTER TABLE memories ADD COLUMN embedding vector(1536)")
    op.execute("ALTER TABLE memories ADD COLUMN embedding_model text")
    op.execute(
        "CREATE INDEX memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("DROP VIEW memories_search")
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
            m.content_tsv,
            m.embedding
        FROM memories m
        JOIN session_memory_stores sm
          ON sm.memory_store_id = m.memory_store_id
         AND sm.session_id = current_setting('app.session_id', true)
        WHERE m.deleted_at IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS memories_search")
    op.execute("DROP INDEX IF EXISTS memories_embedding_hnsw")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS embedding_model")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS embedding")
    op.execute(_VIEW_0119)
