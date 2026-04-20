"""E2E tests for migration 0019 — per-connection routing scope.

Slice 2a covers the structural shape (columns, indexes, FKs) on a
fully-migrated DB.  Slice 2b covers the data-migration path (populating
``connection_id`` + ``path`` from the legacy address strings, dropping
orphans that reference unregistered connections) — it uses its own
dedicated container that stops at 0018, seeds legacy-shape rows, then
runs the 0019 upgrade.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


def _run_alembic(db_url: str, target: str) -> None:
    """Run ``alembic upgrade <target>`` against ``db_url``.

    Subprocess so the alembic config + migration env files control everything.
    """
    env = {**os.environ, "AIOS_DB_URL": db_url}
    result = subprocess.run(
        ["uv", "run", "alembic", "upgrade", target],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade {target} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


@pytest.fixture(scope="module")
def data_migration_db_url() -> Iterator[str]:
    """Fresh container → migrate to 0018 → seed legacy-shape rows →
    migrate to 0019.  Yields the DB URL; tests assert state on the
    result.  Module-scoped so we pay the container/seed cost once;
    the migration is structurally one-way so it doesn't need re-running.
    """
    from tests.conftest import _docker_available

    if not _docker_available():
        pytest.skip("Docker not available")
    import asyncio

    import asyncpg
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        host = pg.get_container_host_ip()
        port = pg.get_exposed_port(5432)
        db_url = f"postgresql://{pg.username}:{pg.password}@{host}:{port}/{pg.dbname}"
        _run_alembic(db_url, "0018")

        async def _seed() -> None:
            c = await asyncpg.connect(db_url)
            try:
                # session_replication_role=replica disables FK triggers for
                # this connection, letting us insert minimal rows referencing
                # stub FK targets without wiring up the full agent/session/vault
                # chain.  The migration we're testing doesn't care about FK
                # validity — only whether it reshapes rules/bindings correctly.
                await c.execute("SET session_replication_role = replica")
                await c.execute("""
                    INSERT INTO connections (id, connector, account, mcp_url, vault_id)
                    VALUES ('conn_sig1', 'signal', 'alice', 'http://mcp-a', 'vlt_stub'),
                           ('conn_sig2', 'signal', 'bob',   'http://mcp-b', 'vlt_stub')
                """)
                # Legacy-shape routing rules:
                #   valid1: exact connector/account match → prefix becomes ''
                #   valid2: connector/account/chat-a      → prefix 'chat-a'
                #   valid3: deeper path (two segments)    → prefix 'group/thread-1'
                #   orphan: unregistered (signal,mallory) → hard-deleted
                await c.execute("""
                    INSERT INTO routing_rules (id, prefix, target, session_params)
                    VALUES
                      ('rul_v1', 'signal/alice',              'session:ses_stub1', '{}'::jsonb),
                      ('rul_v2', 'signal/alice/chat-a',       'session:ses_stub1', '{}'::jsonb),
                      ('rul_v3', 'signal/bob/group/thread-1', 'session:ses_stub2', '{}'::jsonb),
                      ('rul_orph','signal/mallory/chat-z',    'session:ses_stub1', '{}'::jsonb)
                """)
                # Legacy-shape bindings:
                #   valid1: signal/alice/chat-a   → conn_sig1, path 'chat-a'
                #   valid2: signal/bob/grp/thr-2  → conn_sig2, path 'grp/thr-2'
                #   orphan: signal/mallory/chat-z → hard-deleted
                await c.execute("""
                    INSERT INTO channel_bindings (id, address, session_id)
                    VALUES
                      ('bnd_v1',  'signal/alice/chat-a',   'ses_stub1'),
                      ('bnd_v2',  'signal/bob/grp/thr-2',  'ses_stub2'),
                      ('bnd_orph','signal/mallory/chat-z', 'ses_stub1')
                """)
            finally:
                await c.close()

        asyncio.run(_seed())
        _run_alembic(db_url, "head")
        yield db_url


@pytest.fixture
async def mig_pool(data_migration_db_url: str) -> AsyncIterator[Any]:
    import asyncpg

    p = await asyncpg.create_pool(data_migration_db_url, min_size=1, max_size=2)
    try:
        yield p
    finally:
        await p.close()


class TestSchemaShape:
    """Structural assertions on the post-head schema (slice 2a)."""

    async def test_routing_rules_connection_id(self, pool: Any) -> None:
        """``routing_rules.connection_id`` is NOT NULL and FKs to connections."""
        async with pool.acquire() as conn:
            col = await conn.fetchrow(
                """
                SELECT is_nullable
                FROM information_schema.columns
                WHERE table_name = 'routing_rules' AND column_name = 'connection_id'
                """
            )
            assert col is not None, "connection_id column missing on routing_rules"
            assert col["is_nullable"] == "NO"

            fk = await conn.fetchrow(
                """
                SELECT rc.delete_rule
                FROM information_schema.referential_constraints rc
                JOIN information_schema.table_constraints tc
                  ON tc.constraint_name = rc.constraint_name
                WHERE tc.table_name = 'routing_rules'
                  AND rc.delete_rule = 'CASCADE'
                  AND tc.constraint_type = 'FOREIGN KEY'
                  AND EXISTS (
                      SELECT 1 FROM information_schema.key_column_usage kcu
                      WHERE kcu.constraint_name = tc.constraint_name
                        AND kcu.column_name = 'connection_id'
                  )
                """
            )
            assert fk is not None, "routing_rules.connection_id FK missing or not CASCADE"

    async def test_channel_bindings_connection_id_and_path(self, pool: Any) -> None:
        """Bindings gain ``connection_id`` (FK, NOT NULL) and ``path`` (NOT NULL)."""
        async with pool.acquire() as conn:
            cols = await conn.fetch(
                """
                SELECT column_name, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'channel_bindings'
                  AND column_name IN ('connection_id', 'path')
                """
            )
            by_name = {r["column_name"]: r for r in cols}
            assert "connection_id" in by_name, "connection_id missing on channel_bindings"
            assert "path" in by_name, "path missing on channel_bindings"
            assert by_name["connection_id"]["is_nullable"] == "NO"
            assert by_name["path"]["is_nullable"] == "NO"

            fk = await conn.fetchrow(
                """
                SELECT rc.delete_rule
                FROM information_schema.referential_constraints rc
                JOIN information_schema.table_constraints tc
                  ON tc.constraint_name = rc.constraint_name
                WHERE tc.table_name = 'channel_bindings'
                  AND rc.delete_rule = 'CASCADE'
                  AND EXISTS (
                      SELECT 1 FROM information_schema.key_column_usage kcu
                      WHERE kcu.constraint_name = tc.constraint_name
                        AND kcu.column_name = 'connection_id'
                  )
                """
            )
            assert fk is not None, "channel_bindings.connection_id FK missing or not CASCADE"

    async def test_channel_bindings_address_column_dropped(self, pool: Any) -> None:
        """``address`` is normalized away — the full address is computed via JOIN."""
        async with pool.acquire() as conn:
            col = await conn.fetchrow(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'channel_bindings' AND column_name = 'address'
                """
            )
            assert col is None, "address column should be dropped post-0019"

    async def test_indexes_after_migration(self, pool: Any) -> None:
        """Old globally-unique indexes are gone; composite per-connection uniques exist."""
        async with pool.acquire() as conn:
            indexes = await conn.fetch(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename IN ('routing_rules', 'channel_bindings')
                """
            )
            by_name = {r["indexname"]: r["indexdef"] for r in indexes}

            # Old uniques gone
            assert "routing_rules_prefix_uniq" not in by_name
            assert "channel_bindings_address_uniq" not in by_name

            # New composite uniques present (scoped to connection, partial on archived_at)
            rr_idx = by_name.get("routing_rules_conn_prefix_uniq")
            assert rr_idx is not None, "routing_rules_conn_prefix_uniq index missing"
            assert "connection_id" in rr_idx and "prefix" in rr_idx
            assert "archived_at IS NULL" in rr_idx

            cb_idx = by_name.get("channel_bindings_conn_path_uniq")
            assert cb_idx is not None, "channel_bindings_conn_path_uniq index missing"
            assert "connection_id" in cb_idx and "path" in cb_idx
            assert "archived_at IS NULL" in cb_idx


class TestDataMigration:
    """Slice 2b — fixture seeded pre-0019, then migration applied.  All
    assertions read from ``mig_pool`` (a pool against that post-migration
    DB), not the shared ``pool`` fixture.
    """

    async def test_rules_populated_and_rewritten(self, mig_pool: Any) -> None:
        async with mig_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, connection_id, prefix FROM routing_rules ORDER BY id"
            )
            by_id = {r["id"]: (r["connection_id"], r["prefix"]) for r in rows}

            # Exact-match legacy prefix becomes empty (catch-all for the connection).
            assert by_id["rul_v1"] == ("conn_sig1", "")
            # Single extra segment becomes that segment.
            assert by_id["rul_v2"] == ("conn_sig1", "chat-a")
            # Multiple extra segments preserved in the rewritten prefix.
            assert by_id["rul_v3"] == ("conn_sig2", "group/thread-1")

    async def test_orphan_rule_deleted(self, mig_pool: Any) -> None:
        async with mig_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT id FROM routing_rules WHERE id = 'rul_orph'")
            assert row is None, "orphan rule should have been hard-deleted by 0019"

    async def test_bindings_populated_and_address_dropped(self, mig_pool: Any) -> None:
        async with mig_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, connection_id, path FROM channel_bindings ORDER BY id"
            )
            by_id = {r["id"]: (r["connection_id"], r["path"]) for r in rows}
            assert by_id["bnd_v1"] == ("conn_sig1", "chat-a")
            assert by_id["bnd_v2"] == ("conn_sig2", "grp/thr-2")

    async def test_orphan_binding_deleted(self, mig_pool: Any) -> None:
        async with mig_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT id FROM channel_bindings WHERE id = 'bnd_orph'")
            assert row is None, "orphan binding should have been hard-deleted by 0019"

    async def test_post_migration_unique_is_per_connection(self, mig_pool: Any) -> None:
        """Same prefix on different connections is allowed post-0019 —
        the uniqueness is ``(connection_id, prefix)``, not global.
        """
        async with mig_pool.acquire() as conn:
            # Inserting a new rule with the same prefix on conn_sig2 must succeed.
            await conn.execute(
                """
                INSERT INTO routing_rules (id, connection_id, prefix, target, session_params)
                VALUES ('rul_samepfx', 'conn_sig2', 'chat-a', 'session:x', '{}'::jsonb)
                """
            )
            row = await conn.fetchrow(
                "SELECT connection_id, prefix FROM routing_rules WHERE id = 'rul_samepfx'"
            )
            assert row is not None
            assert (row["connection_id"], row["prefix"]) == ("conn_sig2", "chat-a")
