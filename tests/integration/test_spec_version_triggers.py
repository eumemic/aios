"""Integration tests for the ``sessions.spec_version`` bump triggers (#713).

Migration 0077 stamps an integer ``spec_version`` on ``sessions`` and
installs AFTER INSERT/UPDATE/DELETE triggers on the two resource tables
that actually feed :func:`aios.sandbox.spec.build_spec_from_session` —
``session_memory_stores`` and ``session_github_repositories``. Every
mutation of those tables bumps the owning session's ``spec_version`` so
the worker's :class:`aios.sandbox.registry.SandboxRegistry` can detect
drift on a warm hit and recycle the cached sandbox.

Tables that do NOT feed the sandbox spec
(``session_scheduled_tasks``, ``session_vaults``) deliberately have no
bump trigger; two negative tests pin that exclusion and prove no
conflict with migration 0059's NOTIFY trigger on scheduled tasks.

The tests poke the resource tables with raw SQL so each OLD/NEW path of
the trigger is exercised directly, independent of service-layer logic.
Account-scoped parent rows (memory store, vault) are seeded through the
service layer so the column shapes stay correct as the schema evolves.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.services import memory_stores as memory_stores_service
from aios.services import vaults as vaults_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT_ID = "acc_spec_version"


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, session_id)`` for a freshly-seeded session."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ($1, NULL, TRUE, 'spec-version-test')
                """,
                _ACCOUNT_ID,
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT_ID, prefix="spec-version"
        )
        yield pool, session.id
    finally:
        await pool.close()


async def _spec_version(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        value = await conn.fetchval("SELECT spec_version FROM sessions WHERE id = $1", session_id)
    assert value is not None
    return int(value)


async def _attach_memory_store(pool: asyncpg.Pool[Any], session_id: str, store_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO session_memory_stores
                (session_id, memory_store_id, rank, access,
                 name_at_attach, description_at_attach, account_id)
            VALUES ($1, $2, 0, 'read_write', 'notes', 'spec-version store', $3)
            """,
            session_id,
            store_id,
            _ACCOUNT_ID,
        )


# ── memory store triggers ───────────────────────────────────────────────────


async def test_memory_store_attach_bumps_spec_version(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    pool, session_id = pool_and_session
    store = await memory_stores_service.create_store(
        pool, account_id=_ACCOUNT_ID, name="notes", description="d", metadata={}
    )
    before = await _spec_version(pool, session_id)

    await _attach_memory_store(pool, session_id, store.id)

    assert await _spec_version(pool, session_id) == before + 1


async def test_memory_store_delete_bumps_spec_version(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    """DELETE exercises the ``OLD.session_id`` COALESCE branch."""
    pool, session_id = pool_and_session
    store = await memory_stores_service.create_store(
        pool, account_id=_ACCOUNT_ID, name="notes", description="d", metadata={}
    )
    await _attach_memory_store(pool, session_id, store.id)
    before = await _spec_version(pool, session_id)

    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM session_memory_stores WHERE session_id = $1 AND memory_store_id = $2",
            session_id,
            store.id,
        )

    assert await _spec_version(pool, session_id) == before + 1


async def test_memory_store_update_bumps_spec_version(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    pool, session_id = pool_and_session
    store = await memory_stores_service.create_store(
        pool, account_id=_ACCOUNT_ID, name="notes", description="d", metadata={}
    )
    await _attach_memory_store(pool, session_id, store.id)
    before = await _spec_version(pool, session_id)

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE session_memory_stores SET access = 'read_only' "
            "WHERE session_id = $1 AND memory_store_id = $2",
            session_id,
            store.id,
        )

    assert await _spec_version(pool, session_id) == before + 1


# ── github repository triggers ──────────────────────────────────────────────


async def _attach_github_repo(
    pool: asyncpg.Pool[Any], session_id: str, *, mount_path: str = "/mnt/repo"
) -> str:
    repo_id = f"ghrepo_spec_version_{mount_path.strip('/').replace('/', '_')}"
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO session_github_repositories
                (id, session_id, rank, repo_url, mount_path, ciphertext, nonce, account_id)
            VALUES ($1, $2, 0, 'https://github.com/example/repo.git', $3,
                    '\\x00'::bytea, '\\x00'::bytea, $4)
            """,
            repo_id,
            session_id,
            mount_path,
            _ACCOUNT_ID,
        )
    return repo_id


async def test_github_repo_attach_bumps_spec_version(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    pool, session_id = pool_and_session
    before = await _spec_version(pool, session_id)

    await _attach_github_repo(pool, session_id)

    assert await _spec_version(pool, session_id) == before + 1


async def test_github_repo_delete_bumps_spec_version(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    """DELETE exercises the ``OLD.session_id`` COALESCE branch."""
    pool, session_id = pool_and_session
    repo_id = await _attach_github_repo(pool, session_id)
    before = await _spec_version(pool, session_id)

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM session_github_repositories WHERE id = $1", repo_id)

    assert await _spec_version(pool, session_id) == before + 1


async def test_github_repo_update_bumps_spec_version(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    pool, session_id = pool_and_session
    repo_id = await _attach_github_repo(pool, session_id)
    before = await _spec_version(pool, session_id)

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE session_github_repositories SET mount_path = '/mnt/moved' WHERE id = $1",
            repo_id,
        )

    assert await _spec_version(pool, session_id) == before + 1


# ── exclusions: tables that don't feed build_spec_from_session ──────────────


async def test_scheduled_task_insert_does_not_bump_spec_version(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    """Scheduled tasks are read per-turn by the runner, not via
    build_spec_from_session — no bump trigger. Also proves migration
    0077 didn't collide with 0059's NOTIFY trigger on the same table."""
    pool, session_id = pool_and_session
    before = await _spec_version(pool, session_id)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO session_scheduled_tasks
                (id, session_id, account_id, name, schedule, command, enabled,
                 timeout_seconds, max_output_bytes, metadata)
            VALUES ('sched_spec_version_01', $1, $2, 'noop', '* * * * *', 'echo hi',
                    TRUE, 60, 65536, '{}'::jsonb)
            """,
            session_id,
            _ACCOUNT_ID,
        )

    assert await _spec_version(pool, session_id) == before


async def test_vault_binding_insert_bumps_spec_version(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    """Since #873 env-var credentials resolve through a session's bound
    vaults into the provisioning plan, so vault bindings feed the spec
    builder and get a bump trigger (migration 0082) — the Layer-1
    eviction can't cover them (``update_session`` runs in the API
    process, where eviction is a no-op)."""
    pool, session_id = pool_and_session
    vault = await vaults_service.create_vault(
        pool, account_id=_ACCOUNT_ID, display_name="creds", metadata={}
    )
    before = await _spec_version(pool, session_id)

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO session_vaults (session_id, vault_id, rank, account_id) "
            "VALUES ($1, $2, 0, $3)",
            session_id,
            vault.id,
            _ACCOUNT_ID,
        )

    assert await _spec_version(pool, session_id) > before
