"""Integration tests for Layer-1 write-path sandbox eviction (#713).

When a session-scoped resource that feeds
:func:`aios.sandbox.spec.build_spec_from_session` is mutated, the worker
must force the next step to re-provision a fresh sandbox. The mechanism
is :func:`aios.services.sessions._evict_sandbox_for_resource_change`,
which calls ``runtime.sandbox_registry.evict(session_id,
unload_session_caches=False)`` AFTER the mutation transaction commits.

``runtime.sandbox_registry`` is populated only in the WORKER process, so
the helper is a real eviction in-worker and a safe no-op from an API
route (where the global is ``None``). These tests stand in a spy for the
registry global to assert the eviction fires for the right mutations and
stays silent for the ones that don't touch the spec.

Coverage:

- memory-store and github-repository resource changes via
  ``update_session`` evict (they feed the spec).
- vault session-BINDING changes via ``update_session`` evict as
  defense-in-depth (they do NOT feed the spec, but Layer 1 evicts on any
  session-resource mutation).
- a title-only ``update_session`` and ``create_session`` do NOT evict.
- an idempotent re-PUT (same memory resources, same vault ids, or an
  empty list on an empty session) writes nothing: no eviction AND no
  ``spec_version`` bump, so neither layer recycles the sandbox.
- connection attach/detach evict the bound session (defense-in-depth).
- in-place vault credential rotation does NOT evict (the MCP pool keys on
  ``(url, vault_id)`` and the row contents are overwritten in place).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest
from pydantic import SecretStr

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.models.github_repositories import GithubRepositoryResource
from aios.models.memory_stores import MemoryStoreResource
from aios.models.vaults import VaultCredentialCreate
from aios.services import connections as connections_service
from aios.services import memory_stores as memory_stores_service
from aios.services import sessions as sessions_service
from aios.services import vaults as vaults_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT_ID = "acc_evict"


@pytest.fixture
async def env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, CryptoBox]]:
    """Yield ``(pool, session_id, crypto_box)`` for a freshly-seeded session."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    crypto_box = CryptoBox(os.urandom(32))
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ($1, NULL, TRUE, 'evict-test')
                """,
                _ACCOUNT_ID,
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT_ID, prefix="evict"
        )
        yield pool, session.id, crypto_box
    finally:
        await pool.close()


def _install_spy(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stand a spy in for the worker-only registry global.

    ``evict`` is a SYNCHRONOUS method, so the spy uses a plain
    ``MagicMock`` (not ``AsyncMock``) — the production helper must not
    await it.
    """
    spy = MagicMock()
    monkeypatch.setattr(runtime, "sandbox_registry", spy)
    return spy


# ── update_session: resources that feed the spec ────────────────────────────


async def test_update_session_memory_resources_evicts(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session_id, _crypto_box = env
    store = await memory_stores_service.create_store(
        pool, account_id=_ACCOUNT_ID, name="notes", description="d", metadata={}
    )
    spy = _install_spy(monkeypatch)

    await sessions_service.update_session(
        pool,
        session_id,
        account_id=_ACCOUNT_ID,
        resources=[
            MemoryStoreResource(
                type="memory_store",
                memory_store_id=store.id,
                access="read_write",
                instructions="",
            )
        ],
    )

    spy.evict.assert_called_once_with(session_id, unload_session_caches=False)


async def test_update_session_github_resources_evicts(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session_id, crypto_box = env
    spy = _install_spy(monkeypatch)

    await sessions_service.update_session(
        pool,
        session_id,
        account_id=_ACCOUNT_ID,
        resources=[
            GithubRepositoryResource(
                type="github_repository",
                url="https://github.com/example/repo.git",
                mount_path="/mnt/repo",
                authorization_token=SecretStr("ghp_fake_token"),
                git_user_name="u",
                git_user_email="u@example.com",
            )
        ],
        crypto_box=crypto_box,
    )

    spy.evict.assert_called_once_with(session_id, unload_session_caches=False)


async def test_update_session_vault_binding_evicts(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault session-bindings don't feed the spec, but Layer 1 evicts on
    any session-resource mutation as defense-in-depth."""
    pool, session_id, _crypto_box = env
    vault = await vaults_service.create_vault(
        pool, account_id=_ACCOUNT_ID, display_name="creds", metadata={}
    )
    spy = _install_spy(monkeypatch)

    await sessions_service.update_session(
        pool, session_id, account_id=_ACCOUNT_ID, vault_ids=[vault.id]
    )

    spy.evict.assert_called_once_with(session_id, unload_session_caches=False)


# ── update_session / create_session: no resource change ─────────────────────


async def test_update_session_no_resource_or_vault_change_does_not_evict(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session_id, _crypto_box = env
    spy = _install_spy(monkeypatch)

    await sessions_service.update_session(pool, session_id, account_id=_ACCOUNT_ID, title="renamed")

    spy.evict.assert_not_called()


async def test_create_session_does_not_evict(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brand-new session has no cached sandbox to recycle."""
    pool, _session_id, crypto_box = env
    agent, environment, _seed = await seed_agent_env_session(
        pool, account_id=_ACCOUNT_ID, prefix="evict-create"
    )
    store = await memory_stores_service.create_store(
        pool, account_id=_ACCOUNT_ID, name="cs-notes", description="d", metadata={}
    )
    spy = _install_spy(monkeypatch)

    await sessions_service.create_session(
        pool,
        account_id=_ACCOUNT_ID,
        agent_id=agent.id,
        environment_id=environment.id,
        title="fresh",
        metadata={},
        resources=[
            MemoryStoreResource(
                type="memory_store",
                memory_store_id=store.id,
                access="read_write",
                instructions="",
            )
        ],
        crypto_box=crypto_box,
    )

    spy.evict.assert_not_called()


async def test_update_session_idempotent_memory_resources_does_not_evict(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-PUT of the current resource list writes nothing: no Layer-1
    eviction and no Layer-2 ``spec_version`` bump (the triggers fire per
    affected row, and a no-op touches zero rows). Companion to the e2e
    ``test_idempotent_update_does_not_recycle``."""
    pool, session_id, _crypto_box = env
    store = await memory_stores_service.create_store(
        pool, account_id=_ACCOUNT_ID, name="idem-notes", description="d", metadata={}
    )
    resources = [
        MemoryStoreResource(
            type="memory_store",
            memory_store_id=store.id,
            access="read_write",
            instructions="keep",
        )
    ]
    await sessions_service.update_session(
        pool, session_id, account_id=_ACCOUNT_ID, resources=resources
    )
    async with pool.acquire() as conn:
        version_before = await queries.unscoped_get_session_spec_version(conn, session_id)
    spy = _install_spy(monkeypatch)

    await sessions_service.update_session(
        pool, session_id, account_id=_ACCOUNT_ID, resources=resources
    )

    spy.evict.assert_not_called()
    async with pool.acquire() as conn:
        version_after = await queries.unscoped_get_session_spec_version(conn, session_id)
    assert version_after == version_before


async def test_update_session_idempotent_vault_binding_does_not_evict(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session_id, _crypto_box = env
    vault = await vaults_service.create_vault(
        pool, account_id=_ACCOUNT_ID, display_name="idem-creds", metadata={}
    )
    await sessions_service.update_session(
        pool, session_id, account_id=_ACCOUNT_ID, vault_ids=[vault.id]
    )
    spy = _install_spy(monkeypatch)

    await sessions_service.update_session(
        pool, session_id, account_id=_ACCOUNT_ID, vault_ids=[vault.id]
    )

    spy.evict.assert_not_called()


async def test_update_session_empty_resources_on_empty_session_does_not_evict(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resources=[]`` against a session with no attachments detaches
    nothing — no eviction."""
    pool, session_id, _crypto_box = env
    spy = _install_spy(monkeypatch)

    await sessions_service.update_session(pool, session_id, account_id=_ACCOUNT_ID, resources=[])

    spy.evict.assert_not_called()


# ── connections: session-binding changes ────────────────────────────────────


async def test_connection_attach_evicts(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session_id, _crypto_box = env
    async with pool.acquire() as conn:
        connection = await queries.insert_connection(
            conn,
            connector="echo",
            external_account_id="acct-attach",
            metadata={},
            account_id=_ACCOUNT_ID,
        )
    spy = _install_spy(monkeypatch)

    await connections_service.attach_connection(
        pool, connection.id, session_id=session_id, account_id=_ACCOUNT_ID
    )

    spy.evict.assert_called_once_with(session_id, unload_session_caches=False)


async def test_connection_detach_evicts(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session_id, _crypto_box = env
    async with pool.acquire() as conn:
        connection = await queries.insert_connection(
            conn,
            connector="echo",
            external_account_id="acct-detach",
            metadata={},
            account_id=_ACCOUNT_ID,
        )
    # Attach without the spy so only the detach is observed.
    await connections_service.attach_connection(
        pool, connection.id, session_id=session_id, account_id=_ACCOUNT_ID
    )
    spy = _install_spy(monkeypatch)

    await connections_service.detach_connection(pool, connection.id, account_id=_ACCOUNT_ID)

    spy.evict.assert_called_once_with(session_id, unload_session_caches=False)


# ── vaults: in-place credential rotation must NOT evict ─────────────────────


async def test_vault_credential_rotation_does_not_evict(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creating/rotating a credential inside a vault overwrites the row the
    MCP pool already keys on ``(url, vault_id)``; no sandbox eviction."""
    pool, _session_id, crypto_box = env
    vault = await vaults_service.create_vault(
        pool, account_id=_ACCOUNT_ID, display_name="creds", metadata={}
    )
    spy = _install_spy(monkeypatch)

    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=_ACCOUNT_ID,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            target_url="https://mcp.example.com",
            auth_type="bearer_header",
            token=SecretStr("secret-token"),
        ),
    )

    spy.evict.assert_not_called()
