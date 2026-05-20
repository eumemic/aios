"""Integration test: ``clone_session`` must preserve attached memory
stores and github repositories.

Mirrored resources are what makes the clone's next sandbox provision
mount the same shape the parent had at clone time, and what makes
the API response stop lying ``resources: []`` to the caller.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from pydantic import SecretStr

from aios.crypto.vault import CryptoBox
from aios.db.pool import create_pool
from aios.models.github_repositories import GithubRepositoryResource
from aios.models.memory_stores import MemoryStoreResource
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import memory_stores as memory_stores_service
from aios.services import sessions as sessions_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def parent_with_resources(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, CryptoBox]]:
    """Yield ``(pool, account_id, parent_id, crypto_box)`` for an idle
    parent that has one memory store and one github repo attached."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    crypto_box = CryptoBox(os.urandom(32))
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_clone_res', NULL, TRUE, 'clone-resources-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_clone_res",
            name="clone-res-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_clone_res", name="clone-res-env"
        )
        store = await memory_stores_service.create_store(
            pool,
            account_id="acc_clone_res",
            name="notes",
            description="parent's memory store",
            metadata={},
        )
        parent = await sessions_service.create_session(
            pool,
            account_id="acc_clone_res",
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title="parent-with-resources",
            metadata={},
            resources=[
                MemoryStoreResource(
                    type="memory_store",
                    memory_store_id=store.id,
                    access="read_write",
                    instructions="parent attachment",
                ),
                # Two github repos so the ordinal-join's
                # ``ORDER BY rank`` is actually exercised — with one
                # row the ordering is trivial and a rank-scrambling
                # regression would pass silently.
                GithubRepositoryResource(
                    type="github_repository",
                    url="https://github.com/example/first.git",
                    mount_path="/mnt/first",
                    authorization_token=SecretStr("ghp_fake_test_token_1"),
                    git_user_name="test-user",
                    git_user_email="test@example.com",
                ),
                GithubRepositoryResource(
                    type="github_repository",
                    url="https://github.com/example/second.git",
                    mount_path="/mnt/second",
                    authorization_token=SecretStr("ghp_fake_test_token_2"),
                    git_user_name="test-user",
                    git_user_email="test@example.com",
                ),
            ],
            crypto_box=crypto_box,
        )
        # Pin the fixture so a green test can't be due to the parent
        # never having had resources attached.
        assert len(parent.resources) == 3
        memory_echo = next(r for r in parent.resources if r.type == "memory_store")
        assert memory_echo.memory_store_id == store.id
        yield pool, "acc_clone_res", parent.id, crypto_box
    finally:
        await pool.close()


async def test_clone_preserves_memory_and_github_resources(
    parent_with_resources: tuple[asyncpg.Pool[Any], str, str, CryptoBox],
) -> None:
    """Clones of a session with memory + github resources must mirror
    those resources in their own ``Session.resources`` and in the
    underlying ``session_memory_stores`` / ``session_github_repositories``
    tables, so the clone's next sandbox provision mounts what the
    parent had at clone time."""
    pool, account_id, parent_id, _crypto_box = parent_with_resources

    clone = await sessions_service.clone_session(pool, parent_id, account_id=account_id)
    parent = await sessions_service.get_session(pool, parent_id, account_id=account_id)

    clone_memory = [r for r in clone.resources if r.type == "memory_store"]
    clone_github = [r for r in clone.resources if r.type == "github_repository"]
    parent_github = [r for r in parent.resources if r.type == "github_repository"]

    assert len(clone_memory) == 1
    assert clone_memory[0].name == "notes"
    assert clone_memory[0].instructions == "parent attachment"

    # Mount-path order must match the parent's — the ordinal-join in
    # ``clone_session`` keys on ``ORDER BY rank``; a regression that
    # changes the ORDER BY would swap mount paths.
    assert [r.mount_path for r in clone_github] == [r.mount_path for r in parent_github]
    assert [r.url for r in clone_github] == [r.url for r in parent_github]

    # ``session_github_repositories.id`` is a global PK so the clone
    # must mint a fresh ULID per row, not reuse the parent's.
    parent_github_ids = {r.id for r in parent_github}
    for r in clone_github:
        assert r.id.startswith("ghrepo_")
        assert r.id not in parent_github_ids
