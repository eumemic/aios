"""Migration 0196 + hybrid ``memory_search`` (#1629): pgvector as a second RRF signal.

Requires a pgvector-bearing Postgres (the test container is
``pgvector/pgvector:pg16``). The embedder is STUBBED with a deterministic,
hand-authored vectors-by-substring fixture — no live provider in CI.

* **schema** — ``vector`` extension, ``memories.embedding vector(1536)``,
  ``embedding_model``, HNSW index, and the extended view exist; downgrade
  reverses cleanly back to the 0119 view.
* **semantic recall (headline)** — "the customer churned" is surfaced for
  ``memory_search("lost account")`` via the vec arm although
  ``websearch_to_tsquery`` matches nothing.
* **isolation** — the vec arm reads through the same per-session view.
* **NULL embedding** — a row with no embedding still comes back via the FTS arm.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import pytest

from aios.services import memory_embedding as me
from aios.services.memory_embedding import EMBEDDING_DIM, Embedded, EmbedOutcome
from aios.tools.registry import ToolResult
from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root'),
       ('acc_a', 'acc_root', FALSE, 'tenant-a'),
       ('acc_b', 'acc_root', FALSE, 'tenant-b');
INSERT INTO agents (id, account_id, name, version, model, system, window_min, window_max)
VALUES ('agent_a', 'acc_a', 'a', 1, 'openrouter/test', '', 50000, 150000),
       ('agent_b', 'acc_b', 'b', 1, 'openrouter/test', '', 50000, 150000);
INSERT INTO environments (id, name, config, account_id)
VALUES ('env_a', 'env-a', '{}'::jsonb, 'acc_a'),
       ('env_b', 'env-b', '{}'::jsonb, 'acc_b');
INSERT INTO sessions
    (id, account_id, agent_id, environment_id, agent_version, workspace_volume_path)
VALUES ('sess_x', 'acc_a', 'agent_a', 'env_a', 1, '/tmp/sess_x'),
       ('sess_y', 'acc_b', 'agent_b', 'env_b', 1, '/tmp/sess_y');
INSERT INTO memory_stores (id, name, account_id)
VALUES ('store_s1', 'store-s1', 'acc_a'),
       ('store_s2', 'store-s2', 'acc_b');
INSERT INTO session_memory_stores
    (session_id, memory_store_id, rank, access,
     name_at_attach, description_at_attach, account_id)
VALUES ('sess_x', 'store_s1', 0, 'read_write', 'store-s1', '', 'acc_a'),
       ('sess_y', 'store_s2', 0, 'read_write', 'store-s2', '', 'acc_b');
"""


def _axis(i: int, j: int | None = None) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    if j is not None:
        v[j] = 0.2
    return v


# Hand-authored vectors-by-substring: "churned" and "lost account" are close
# (both on axis 0); everything else points elsewhere.
_FIXTURE: list[tuple[str, list[float]]] = [
    ("churned", _axis(0, 1)),
    ("lost account", _axis(0, 2)),
    ("deploy", _axis(5)),
]


async def _stub_embed(content: str) -> EmbedOutcome:
    for needle, vec in _FIXTURE:
        if needle in content:
            return Embedded(vector=vec, model="stub-embed")
    return Embedded(vector=_axis(9), model="stub-embed")


def _insert_sql(mem_id: str, store_id: str, path: str, content: str, account_id: str) -> str:
    return (
        "INSERT INTO memories (id, memory_store_id, path, content, content_sha256, "
        " content_size_bytes, account_id) "
        f"VALUES ('{mem_id}', '{store_id}', '{path}', $${content}$$, 'sha_{mem_id}', "
        f"octet_length($${content}$$), '{account_id}')"
    )


async def _execute(db_url: str, sql: str, *args: object) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql, *args)
    finally:
        await conn.close()


async def _fetch(db_url: str, sql: str, *args: object) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(db_url)
    try:
        return list(await conn.fetch(sql, *args))
    finally:
        await conn.close()


async def _embed_row(db_url: str, mem_id: str, account_id: str, content: str) -> bool:
    from aios.db import queries

    outcome = await _stub_embed(content)
    assert isinstance(outcome, Embedded)
    conn = await asyncpg.connect(db_url)
    try:
        return await queries.update_memory_embedding(
            conn,
            account_id=account_id,
            memory_id=mem_id,
            embedding=outcome.vector,
            embedding_model=outcome.model,
            content_sha256=f"sha_{mem_id}",
        )
    finally:
        await conn.close()


def _search(db_url: str, session_id: str, query: str) -> str:
    async def _run() -> ToolResult:
        from aios.db.pool import create_pool
        from aios.harness import runtime
        from aios.tools.memory_search import memory_search_handler

        pool = await create_pool(db_url, min_size=1, max_size=2)
        prev = runtime.pool
        runtime.pool = pool
        try:
            return await memory_search_handler(session_id, {"query": query})
        finally:
            runtime.pool = prev
            await pool.close()

    result = asyncio.run(_run())
    assert result.is_error is False
    assert isinstance(result.content, str)
    return result.content


@pytest.fixture
def stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    import aios.tools.memory_search as ms

    monkeypatch.setattr(ms, "embed_memory_content", _stub_embed)
    monkeypatch.setattr(me, "embed_memory_content", _stub_embed)


def _seed(db_url: str, rows: list[tuple[str, str, str, str, str]]) -> None:
    assert run_alembic(["upgrade", "head"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))
    for row in rows:
        asyncio.run(_execute(db_url, _insert_sql(*row)))


@needs_docker
@pytest.mark.integration
def test_schema_and_downgrade(migration_db_url: str) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "head"], db_url).returncode == 0
    ext = asyncio.run(_fetch(db_url, "SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
    assert len(ext) == 1
    col = asyncio.run(
        _fetch(
            db_url,
            "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
            "WHERE attrelid = 'memories'::regclass AND attname = 'embedding'",
        )
    )
    assert col[0]["t"] == f"vector({EMBEDDING_DIM})"
    idx = asyncio.run(
        _fetch(
            db_url,
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'memories_embedding_hnsw'",
        )
    )
    assert "hnsw" in idx[0]["indexdef"].lower()
    view_cols = asyncio.run(
        _fetch(
            db_url,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'memories_search'",
        )
    )
    assert "embedding" in {r["column_name"] for r in view_cols}

    assert run_alembic(["downgrade", "0195"], db_url).returncode == 0
    gone = asyncio.run(
        _fetch(
            db_url,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name IN ('memories', 'memories_search') "
            "AND column_name IN ('embedding', 'embedding_model')",
        )
    )
    assert gone == []
    view = asyncio.run(
        _fetch(
            db_url, "SELECT 1 FROM information_schema.views WHERE table_name = 'memories_search'"
        )
    )
    assert len(view) == 1, "0119 view not restored on downgrade"
    assert run_alembic(["upgrade", "head"], db_url).returncode == 0


@needs_docker
@pytest.mark.integration
def test_semantic_recall_via_vec_arm(migration_db_url: str, stub_embedder: None) -> None:
    db_url = migration_db_url
    content = "the customer churned"
    _seed(db_url, [("mem_c", "store_s1", "/churn.md", content, "acc_a")])
    assert asyncio.run(_embed_row(db_url, "mem_c", "acc_a", content))

    # FTS alone finds nothing — the gap #1372 cannot close.
    fts = asyncio.run(
        _fetch(
            db_url,
            "SELECT id FROM memories WHERE content_tsv @@ websearch_to_tsquery('english', $1)",
            "lost account",
        )
    )
    assert fts == []

    text = _search(db_url, "sess_x", "lost account")
    assert "/churn.md" in text


@needs_docker
@pytest.mark.integration
def test_vec_arm_respects_session_isolation(migration_db_url: str, stub_embedder: None) -> None:
    db_url = migration_db_url
    content = "the customer churned"
    _seed(db_url, [("mem_c", "store_s1", "/churn.md", content, "acc_a")])
    assert asyncio.run(_embed_row(db_url, "mem_c", "acc_a", content))
    assert "/churn.md" not in _search(db_url, "sess_y", "lost account")
    assert _search(db_url, "sess_y", "lost account") == "No results."


@needs_docker
@pytest.mark.integration
def test_null_embedding_row_found_by_fts(migration_db_url: str, stub_embedder: None) -> None:
    db_url = migration_db_url
    _seed(
        db_url,
        [
            ("mem_n", "store_s1", "/null.md", "deploy rollback procedure", "acc_a"),
            ("mem_e", "store_s1", "/emb.md", "unrelated gardening notes", "acc_a"),
        ],
    )
    assert asyncio.run(_embed_row(db_url, "mem_e", "acc_a", "unrelated gardening notes"))
    rows = asyncio.run(_fetch(db_url, "SELECT embedding FROM memories WHERE id = 'mem_n'"))
    assert rows[0]["embedding"] is None
    text = _search(db_url, "sess_x", "deploy")
    assert "/null.md" in text


@needs_docker
@pytest.mark.integration
def test_query_embed_failure_degrades_to_fts_only(
    migration_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aios.tools.memory_search as ms

    db_url = migration_db_url
    _seed(
        db_url,
        [
            ("mem_k", "store_s1", "/kw.md", "deploy rollback", "acc_a"),
            ("mem_v", "store_s1", "/vec.md", "the customer churned", "acc_a"),
        ],
    )
    assert asyncio.run(_embed_row(db_url, "mem_v", "acc_a", "the customer churned"))

    async def _down(content: str) -> Any:
        return me.Skipped("provider down")

    monkeypatch.setattr(ms, "embed_memory_content", _down)
    text = _search(db_url, "sess_x", "deploy")
    assert "/kw.md" in text
    assert "/vec.md" not in text


@needs_docker
@pytest.mark.integration
def test_stale_embedding_write_is_noop(migration_db_url: str) -> None:
    """The sha guard drops an embedding computed for superseded content."""
    from aios.db import queries

    db_url = migration_db_url
    _seed(db_url, [("mem_s", "store_s1", "/s.md", "x", "acc_a")])

    async def _write() -> bool:
        conn = await asyncpg.connect(db_url)
        try:
            return await queries.update_memory_embedding(
                conn,
                account_id="acc_a",
                memory_id="mem_s",
                embedding=_axis(1),
                embedding_model="stub-embed",
                content_sha256="stale",
            )
        finally:
            await conn.close()

    assert asyncio.run(_write()) is False
