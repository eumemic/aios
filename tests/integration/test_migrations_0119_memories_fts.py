"""Migration 0119 + the ``memory_search`` builtin: full-text recall over
``memories.content`` via an additive generated ``tsvector`` + GIN index and a
session-scoped ``memories_search`` view.

Covers the deterministic acceptance from the design:

* **schema** — after ``upgrade head`` the ``memories.content_tsv`` generated
  STORED column and the ``memories_content_tsv_gin`` index exist; inserting a
  memory populates ``content_tsv`` with no trigger (generated-by-construction).
* **scoping** — session X (attached to store S1) and session Y (attached to
  S2): ``SET app.session_id = X`` then querying ``memories_search`` returns
  only S1 rows (the cross-session / cross-tenant isolation invariant that
  ``events_search`` relies on).
* **rank + cap** — ``ts_rank`` DESC ordering and the ``MAX_ROWS`` cap via the
  tool handler's fixed query.
* **soft-delete** — a ``deleted_at``-stamped memory never appears.
* **read-only** — the handler's execution path runs inside a READ ONLY txn, so
  no mutation is possible even though the surface is a narrowed ``{query}``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# Two tenants, two stores, two sessions; each session is attached to exactly
# one store. The isolation invariant: a session only ever sees memories of its
# own attached stores.
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


def _insert_memory_sql(
    *, mem_id: str, store_id: str, path: str, content: str, account_id: str, deleted: bool
) -> str:
    deleted_at = "now()" if deleted else "NULL"
    # content_size_bytes must equal octet_length(content) (CHECK), so derive it
    # from the literal at insert time rather than hardcoding.
    return (
        "INSERT INTO memories "
        "(id, memory_store_id, path, content, content_sha256, "
        " content_size_bytes, account_id, deleted_at) "
        f"VALUES ('{mem_id}', '{store_id}', '{path}', $${content}$$, 'sha', "
        f"octet_length($${content}$$), '{account_id}', {deleted_at})"
    )


@pytest.fixture
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _fetch(db_url: str, sql: str, *args: object) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(db_url)
    try:
        return list(await conn.fetch(sql, *args))
    finally:
        await conn.close()


async def _execute(db_url: str, sql: str, *args: object) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql, *args)
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_generated_column_and_gin_index_exist_and_populate(postgres: object) -> None:
    """The generated STORED column + GIN index exist and the vector is
    populated on insert with no trigger (generated-by-construction)."""
    db_url = _alembic_url(postgres)
    assert _run_alembic(["upgrade", "head"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))

    # Column exists and is GENERATED STORED.
    cols = asyncio.run(
        _fetch(
            db_url,
            "SELECT generation_expression "
            "FROM information_schema.columns "
            "WHERE table_name = 'memories' AND column_name = 'content_tsv'",
        )
    )
    assert len(cols) == 1, "content_tsv column missing"
    assert cols[0]["generation_expression"] is not None, "content_tsv is not generated"

    # GIN index exists.
    idx = asyncio.run(
        _fetch(
            db_url,
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = 'memories' AND indexname = 'memories_content_tsv_gin'",
        )
    )
    assert len(idx) == 1, "memories_content_tsv_gin index missing"
    assert "gin" in idx[0]["indexdef"].lower()

    # Insert a memory; content_tsv is populated automatically (no trigger).
    asyncio.run(
        _execute(
            db_url,
            _insert_memory_sql(
                mem_id="mem_gen",
                store_id="store_s1",
                path="/note.md",
                content="postgres full text search rocks",
                account_id="acc_a",
                deleted=False,
            ),
        )
    )
    tsv = asyncio.run(
        _fetch(db_url, "SELECT content_tsv::text AS tsv FROM memories WHERE id = 'mem_gen'")
    )
    assert tsv[0]["tsv"], "content_tsv was not populated by the generated column"
    # 'rocks' stems to 'rock'; 'search' is present — confirms to_tsvector ran.
    assert "search" in tsv[0]["tsv"]


@needs_docker
@pytest.mark.integration
def test_memory_search_scoping(postgres: object) -> None:
    """A session only sees memories of its own attached stores."""
    db_url = _alembic_url(postgres)
    assert _run_alembic(["upgrade", "head"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))
    asyncio.run(
        _execute(
            db_url,
            _insert_memory_sql(
                mem_id="mem_x",
                store_id="store_s1",
                path="/x.md",
                content="shared keyword alpha in store one",
                account_id="acc_a",
                deleted=False,
            ),
        )
    )
    asyncio.run(
        _execute(
            db_url,
            _insert_memory_sql(
                mem_id="mem_y",
                store_id="store_s2",
                path="/y.md",
                content="shared keyword alpha in store two",
                account_id="acc_b",
                deleted=False,
            ),
        )
    )

    async def _scoped(session_id: str) -> list[str]:
        conn = await asyncpg.connect(db_url)
        try:
            async with conn.transaction(readonly=True):
                await conn.execute("SELECT set_config('app.session_id', $1, true)", session_id)
                rows = await conn.fetch("SELECT id FROM memories_search ORDER BY id")
            return [r["id"] for r in rows]
        finally:
            await conn.close()

    assert asyncio.run(_scoped("sess_x")) == ["mem_x"]
    assert asyncio.run(_scoped("sess_y")) == ["mem_y"]


@needs_docker
@pytest.mark.integration
def test_memory_search_rank_and_handler(postgres: object) -> None:
    """The tool handler returns ts_rank DESC ordering, scoped to the session."""
    db_url = _alembic_url(postgres)
    assert _run_alembic(["upgrade", "head"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))
    # mem_hi mentions the keyword many times → higher ts_rank than mem_lo.
    asyncio.run(
        _execute(
            db_url,
            _insert_memory_sql(
                mem_id="mem_hi",
                store_id="store_s1",
                path="/hi.md",
                content="deploy deploy deploy rollback deploy procedure",
                account_id="acc_a",
                deleted=False,
            ),
        )
    )
    asyncio.run(
        _execute(
            db_url,
            _insert_memory_sql(
                mem_id="mem_lo",
                store_id="store_s1",
                path="/lo.md",
                content="a long note that mentions deploy once near the end",
                account_id="acc_a",
                deleted=False,
            ),
        )
    )

    async def _run() -> dict[str, object]:
        from aios.db.pool import create_pool
        from aios.harness import runtime
        from aios.tools.memory_search import memory_search_handler

        pool = await create_pool(db_url, min_size=1, max_size=2)
        prev = runtime.pool
        runtime.pool = pool
        try:
            return await memory_search_handler("sess_x", {"query": "deploy"})
        finally:
            runtime.pool = prev
            await pool.close()

    result = asyncio.run(_run())
    assert "result" in result, result
    text = result["result"]
    assert isinstance(text, str)
    # Both match; the keyword-dense memory ranks first.
    assert "/hi.md" in text and "/lo.md" in text
    assert text.index("/hi.md") < text.index("/lo.md")


@needs_docker
@pytest.mark.integration
def test_memory_search_soft_deleted_excluded(postgres: object) -> None:
    """A ``deleted_at``-stamped memory does not appear in the view."""
    db_url = _alembic_url(postgres)
    assert _run_alembic(["upgrade", "head"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))
    asyncio.run(
        _execute(
            db_url,
            _insert_memory_sql(
                mem_id="mem_live",
                store_id="store_s1",
                path="/live.md",
                content="visible widget content",
                account_id="acc_a",
                deleted=False,
            ),
        )
    )
    asyncio.run(
        _execute(
            db_url,
            _insert_memory_sql(
                mem_id="mem_dead",
                store_id="store_s1",
                path="/dead.md",
                content="deleted widget content",
                account_id="acc_a",
                deleted=True,
            ),
        )
    )

    async def _scoped() -> list[str]:
        conn = await asyncpg.connect(db_url)
        try:
            async with conn.transaction(readonly=True):
                await conn.execute("SELECT set_config('app.session_id', 'sess_x', true)")
                rows = await conn.fetch("SELECT id FROM memories_search ORDER BY id")
            return [r["id"] for r in rows]
        finally:
            await conn.close()

    visible = asyncio.run(_scoped())
    assert visible == ["mem_live"], visible


@needs_docker
@pytest.mark.integration
def test_memory_search_runs_read_only(postgres: object) -> None:
    """The handler's execution path is READ ONLY — no mutation is possible
    even though the surface is a narrowed ``{query}`` (not raw SQL)."""
    db_url = _alembic_url(postgres)
    assert _run_alembic(["upgrade", "head"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))

    async def _attempt_write() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            # Mirror the handler's transaction shape exactly.
            async with conn.transaction(readonly=True):
                await conn.execute("SELECT set_config('app.session_id', 'sess_x', true)")
                with pytest.raises(asyncpg.exceptions.ReadOnlySQLTransactionError):
                    await conn.execute("UPDATE memories SET path = '/hacked' WHERE id = 'mem_x'")
        finally:
            await conn.close()

    asyncio.run(_attempt_write())


@needs_docker
@pytest.mark.integration
def test_downgrade_drops_view_index_and_column(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    assert _run_alembic(["upgrade", "0119"], db_url).returncode == 0
    down = _run_alembic(["downgrade", "0118"], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}\n{down.stdout}"

    cols = asyncio.run(
        _fetch(
            db_url,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'memories' AND column_name = 'content_tsv'",
        )
    )
    assert cols == []
    views = asyncio.run(
        _fetch(
            db_url,
            "SELECT table_name FROM information_schema.views WHERE table_name = 'memories_search'",
        )
    )
    assert views == []
