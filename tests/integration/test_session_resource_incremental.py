"""Integration tests for granular add-one / remove-one session resources (#270).

The full-list-replace ``PUT /v1/sessions/{id}`` silently detaches every
resource the caller omits. These tests cover the additive sub-collection
service surface — :func:`aios.services.sessions.add_resource` /
:func:`aios.services.sessions.remove_resource` — that lets an operator
mutate ONE resource without re-supplying the rest:

- add-one (github + memory) leaves the other attachments intact (the #270
  regression).
- a colliding memory name is rejected with a single attachment left
  standing (blocker-1: no silent dual-mount).
- fill-to-cap → delete a low rank → add-one stays inside the
  ``CHECK (rank BETWEEN 0 AND 7)`` bound (blocker-2: lowest-free-rank).
- N concurrent add-ones serialize on the per-session advisory lock; the
  cap is never overshot (cap is contractual).
- remove-one detaches exactly the targeted attachment for both types;
  github working-tree purge fires; memory_versions are untouched.
- a mount_path collision on github add-one is a 4xx.
- a cross-tenant session id is a 404 (queries are account-scoped).
- a malformed / unknown-prefix id is a 4xx ValidationError, not 404/500.
"""

from __future__ import annotations

import asyncio
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
from aios.errors import ConflictError, NotFoundError, RateLimitedError, ValidationError
from aios.models.github_repositories import (
    MAX_REPOS_PER_SESSION,
    GithubRepositoryResource,
    GithubRepositoryResourceEcho,
)
from aios.models.memory_stores import MAX_STORES_PER_SESSION, MemoryStoreResource
from aios.services import memory_stores as memory_stores_service
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT_ID = "acc_incr"
_OTHER_ACCOUNT_ID = "acc_incr_other"


@pytest.fixture
async def env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, CryptoBox]]:
    """Yield ``(pool, session_id, crypto_box)`` for a freshly-seeded session."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=8)
    crypto_box = CryptoBox(os.urandom(32))
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ($1, NULL, TRUE, 'incr-test')
                """,
                _ACCOUNT_ID,
            )
            # The OTHER tenant is a child of the root — only one active root
            # is allowed (``accounts_one_active_root``). A sibling tenant is
            # all the cross-tenant isolation tests need.
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ($1, $2, FALSE, 'incr-other')
                """,
                _OTHER_ACCOUNT_ID,
                _ACCOUNT_ID,
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT_ID, prefix="incr"
        )
        yield pool, session.id, crypto_box
    finally:
        await pool.close()


def _gh(url: str, mount_path: str, token: str = "ghp_fake") -> GithubRepositoryResource:
    return GithubRepositoryResource(
        type="github_repository",
        url=url,
        mount_path=mount_path,
        authorization_token=SecretStr(token),
    )


def _mem(store_id: str) -> MemoryStoreResource:
    return MemoryStoreResource(
        type="memory_store",
        memory_store_id=store_id,
        access="read_write",
        instructions="",
    )


async def _make_store(pool: asyncpg.Pool[Any], name: str, *, account_id: str = _ACCOUNT_ID) -> str:
    store = await memory_stores_service.create_store(
        pool, account_id=account_id, name=name, description="d", metadata={}
    )
    return store.id


async def _add_gh(
    pool: asyncpg.Pool[Any],
    session_id: str,
    url: str,
    mount_path: str,
    crypto_box: CryptoBox,
) -> GithubRepositoryResourceEcho:
    """Add a github resource and narrow the union echo to the github type
    so callers can read ``.id`` without a mypy union-attr error."""
    echo = await sessions_service.add_resource(
        pool, session_id, _gh(url, mount_path), crypto_box=crypto_box, account_id=_ACCOUNT_ID
    )
    assert isinstance(echo, GithubRepositoryResourceEcho)
    return echo


# ── criterion 1: github add-one leaves existing resources attached ──────────


async def test_add_github_does_not_detach_existing(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, crypto_box = env
    store_id = await _make_store(pool, "ultron-memory")
    # Seed one memory store + one github repo via the full-list PUT.
    await sessions_service.update_session(
        pool,
        session_id,
        account_id=_ACCOUNT_ID,
        resources=[_mem(store_id), _gh("https://x/a.git", "/mnt/a")],
        crypto_box=crypto_box,
    )

    echo = await sessions_service.add_resource(
        pool,
        session_id,
        _gh("https://x/b.git", "/mnt/b"),
        crypto_box=crypto_box,
        account_id=_ACCOUNT_ID,
    )

    assert echo.type == "github_repository"
    assert echo.mount_path == "/mnt/b"
    async with pool.acquire() as conn:
        mem = await queries.list_session_memory_store_echoes(
            conn, session_id, account_id=_ACCOUNT_ID
        )
        gh = await queries.list_session_github_repo_echoes(conn, session_id, account_id=_ACCOUNT_ID)
    assert [m.memory_store_id for m in mem] == [store_id]  # memory store still attached
    assert sorted(g.mount_path for g in gh) == ["/mnt/a", "/mnt/b"]


# ── criterion 2: memory add-one appends with derived mount_path ─────────────


async def test_add_memory_appends_with_mount_path(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, _crypto_box = env
    first = await _make_store(pool, "first")
    second = await _make_store(pool, "second")
    await sessions_service.add_resource(
        pool, session_id, _mem(first), crypto_box=_crypto_box, account_id=_ACCOUNT_ID
    )

    echo = await sessions_service.add_resource(
        pool, session_id, _mem(second), crypto_box=_crypto_box, account_id=_ACCOUNT_ID
    )

    assert echo.type == "memory_store"
    assert echo.mount_path == "/mnt/memory/second"
    async with pool.acquire() as conn:
        mem = await queries.list_session_memory_store_echoes(
            conn, session_id, account_id=_ACCOUNT_ID
        )
    assert sorted(m.name for m in mem) == ["first", "second"]


# ── criterion 3 (blocker-1): colliding memory name → 409, one attachment ────


async def test_add_memory_name_collision_rejected_single_attachment(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, _crypto_box = env
    a = await _make_store(pool, "dup")
    b = await _make_store(pool, "dup")  # different store, same name
    await sessions_service.add_resource(
        pool, session_id, _mem(a), crypto_box=_crypto_box, account_id=_ACCOUNT_ID
    )

    with pytest.raises(ConflictError):
        await sessions_service.add_resource(
            pool, session_id, _mem(b), crypto_box=_crypto_box, account_id=_ACCOUNT_ID
        )

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM session_memory_stores WHERE session_id = $1 AND name_at_attach = 'dup'",
            session_id,
        )
    assert len(rows) == 1  # no silent dual-mount


# ── criterion 5 (blocker-2): fill, delete a low rank, add → lowest-free rank ─


async def test_fill_delete_low_rank_then_add_stays_in_bound(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, _crypto_box = env
    store_ids = [await _make_store(pool, f"s{i}") for i in range(MAX_STORES_PER_SESSION)]
    for sid in store_ids:
        await sessions_service.add_resource(
            pool, session_id, _mem(sid), crypto_box=_crypto_box, account_id=_ACCOUNT_ID
        )
    # Delete a low-rank attachment (rank 3 by insertion order).
    await sessions_service.remove_resource(pool, session_id, store_ids[3], account_id=_ACCOUNT_ID)

    extra = await _make_store(pool, "extra")
    # Would 500 on a CHECK violation if rank were max+1 == 8.
    echo = await sessions_service.add_resource(
        pool, session_id, _mem(extra), crypto_box=_crypto_box, account_id=_ACCOUNT_ID
    )
    assert echo.type == "memory_store"
    async with pool.acquire() as conn:
        ranks = await queries.list_session_memory_store_ranks(
            conn, session_id, account_id=_ACCOUNT_ID
        )
    assert max(ranks) <= MAX_STORES_PER_SESSION - 1
    assert 3 in ranks  # the freed slot was reused


# ── criterion 6: over-cap add → 4xx ─────────────────────────────────────────


async def test_add_memory_over_cap_rejected(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, _crypto_box = env
    for i in range(MAX_STORES_PER_SESSION):
        sid = await _make_store(pool, f"cap{i}")
        await sessions_service.add_resource(
            pool, session_id, _mem(sid), crypto_box=_crypto_box, account_id=_ACCOUNT_ID
        )
    over = await _make_store(pool, "over")
    with pytest.raises(RateLimitedError):
        await sessions_service.add_resource(
            pool, session_id, _mem(over), crypto_box=_crypto_box, account_id=_ACCOUNT_ID
        )


async def test_add_github_over_cap_rejected(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, crypto_box = env
    for i in range(MAX_REPOS_PER_SESSION):
        await sessions_service.add_resource(
            pool,
            session_id,
            _gh(f"https://x/{i}.git", f"/mnt/r{i}"),
            crypto_box=crypto_box,
            account_id=_ACCOUNT_ID,
        )
    with pytest.raises(RateLimitedError):
        await sessions_service.add_resource(
            pool,
            session_id,
            _gh("https://x/over.git", "/mnt/over"),
            crypto_box=crypto_box,
            account_id=_ACCOUNT_ID,
        )


# ── criterion 7: concurrent add-ones serialize; cap never overshot ──────────


async def test_concurrent_adds_never_overshoot_cap(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, crypto_box = env
    # Pre-fill to one below the cap, then fire N concurrent adds at the boundary.
    for i in range(MAX_REPOS_PER_SESSION - 1):
        await sessions_service.add_resource(
            pool,
            session_id,
            _gh(f"https://x/pre{i}.git", f"/mnt/pre{i}"),
            crypto_box=crypto_box,
            account_id=_ACCOUNT_ID,
        )

    async def _add(n: int) -> None:
        await sessions_service.add_resource(
            pool,
            session_id,
            _gh(f"https://x/c{n}.git", f"/mnt/c{n}"),
            crypto_box=crypto_box,
            account_id=_ACCOUNT_ID,
        )

    results = await asyncio.gather(*[_add(n) for n in range(5)], return_exceptions=True)
    # Exactly one add should land (filling the last slot); the rest hit the cap.
    succeeded = [r for r in results if not isinstance(r, BaseException)]
    capped = [r for r in results if isinstance(r, RateLimitedError)]
    assert len(succeeded) == 1
    assert len(capped) == 4
    async with pool.acquire() as conn:
        gh = await queries.list_session_github_repo_echoes(conn, session_id, account_id=_ACCOUNT_ID)
    assert len(gh) == MAX_REPOS_PER_SESSION  # never overshot


# ── criterion 8: remove github by id, others intact, working tree purged ────


async def test_remove_github_by_id(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session_id, crypto_box = env
    keep = await _add_gh(pool, session_id, "https://x/keep.git", "/mnt/keep", crypto_box)
    drop = await _add_gh(pool, session_id, "https://x/drop.git", "/mnt/drop", crypto_box)
    purge = MagicMock()
    monkeypatch.setattr("aios.services.github_repositories.remove_session_working_tree", purge)

    await sessions_service.remove_resource(pool, session_id, drop.id, account_id=_ACCOUNT_ID)

    purge.assert_called_once_with(session_id, drop.id)
    async with pool.acquire() as conn:
        gh = await queries.list_session_github_repo_echoes(conn, session_id, account_id=_ACCOUNT_ID)
    assert [g.id for g in gh] == [keep.id]


# ── criterion 9: remove memory by id, others intact, versions untouched ─────


async def test_remove_memory_by_id_versions_untouched(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, _crypto_box = env
    keep = await _make_store(pool, "keep")
    drop = await _make_store(pool, "drop")
    # Write a memory version on the store we'll detach.
    await memory_stores_service.create_memory(
        pool,
        store_id=drop,
        path="/notes.md",
        content="hello",
        actor=memory_stores_service.ApiActor(),
        account_id=_ACCOUNT_ID,
    )
    for sid in (keep, drop):
        await sessions_service.add_resource(
            pool, session_id, _mem(sid), crypto_box=_crypto_box, account_id=_ACCOUNT_ID
        )

    await sessions_service.remove_resource(pool, session_id, drop, account_id=_ACCOUNT_ID)

    async with pool.acquire() as conn:
        mem = await queries.list_session_memory_store_echoes(
            conn, session_id, account_id=_ACCOUNT_ID
        )
        versions = await conn.fetchval(
            "SELECT count(*) FROM memory_versions WHERE memory_store_id = $1", drop
        )
    assert [m.memory_store_id for m in mem] == [keep]
    assert versions == 1  # never-delete: the version survives detach


# ── criterion 4: github mount_path collision → ConflictError ────────────────


async def test_add_github_mount_path_collision(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, crypto_box = env
    await sessions_service.add_resource(
        pool,
        session_id,
        _gh("https://x/a.git", "/mnt/same"),
        crypto_box=crypto_box,
        account_id=_ACCOUNT_ID,
    )
    with pytest.raises(ConflictError):
        await sessions_service.add_resource(
            pool,
            session_id,
            _gh("https://x/b.git", "/mnt/same"),
            crypto_box=crypto_box,
            account_id=_ACCOUNT_ID,
        )


# ── criterion 10: malformed / unknown-prefix id → ValidationError ───────────


async def test_remove_malformed_id_raises_validation(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, _crypto_box = env
    with pytest.raises(ValidationError):
        await sessions_service.remove_resource(
            pool, session_id, "not-a-valid-id", account_id=_ACCOUNT_ID
        )


async def test_remove_unknown_prefix_id_raises_validation(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, _crypto_box = env
    with pytest.raises(ValidationError):
        await sessions_service.remove_resource(
            pool, session_id, "trig_01HQR2K7VXBZ9MNPL3WYCT8F00", account_id=_ACCOUNT_ID
        )


# ── criterion 11: cross-tenant session id → 404 ─────────────────────────────


async def test_add_cross_tenant_session_not_found(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, crypto_box = env
    store_id = await _make_store(pool, "x")
    # Same session_id but the OTHER account: queries are account-scoped, so
    # the store fetch under the wrong account is a 404.
    other_store = await _make_store(pool, "other", account_id=_OTHER_ACCOUNT_ID)
    with pytest.raises(NotFoundError):
        await sessions_service.add_resource(
            pool,
            session_id,
            _mem(other_store),
            crypto_box=crypto_box,
            account_id=_OTHER_ACCOUNT_ID,
        )
    assert store_id  # silence unused (the in-tenant store exists for contrast)


async def test_remove_cross_tenant_not_found(
    env: tuple[asyncpg.Pool[Any], str, CryptoBox],
) -> None:
    pool, session_id, crypto_box = env
    gh = await _add_gh(pool, session_id, "https://x/a.git", "/mnt/a", crypto_box)
    with pytest.raises(NotFoundError):
        await sessions_service.remove_resource(
            pool, session_id, gh.id, account_id=_OTHER_ACCOUNT_ID
        )
