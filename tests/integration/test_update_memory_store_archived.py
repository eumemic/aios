"""Integration test: ``queries.update_memory_store`` must refuse to
rewrite an archived store.

The service-layer ``update_store`` (services/memory_stores.py:134)
already pre-fetches and raises ``MemoryStoreArchivedError`` on
archived rows, but the query-layer ``update_memory_store`` has no
guard.  Same defect-class as PR #547 (update_session_template), PR
#554 (update_vault), PR #573 (update_session) — silent-write-on-
archived-row family.  Closes the last named sibling from PR #554's
commit body.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import MemoryStoreArchivedError

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_memory_store(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, store_id)`` for a memory store that
    has been archived after creation."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_mem_arch', NULL, TRUE, 'memstore-archived-test')
                """
            )
            store = await queries.insert_memory_store(
                conn,
                name="pre-archive",
                description="the parent description",
                metadata={},
                account_id="acc_mem_arch",
            )
            archived = await queries.archive_memory_store(conn, store.id, account_id="acc_mem_arch")
        assert archived.archived_at is not None
        yield pool, "acc_mem_arch", store.id
    finally:
        await pool.close()


async def test_update_memory_store_refuses_archived(
    archived_memory_store: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A direct call to ``queries.update_memory_store`` on an archived
    row must raise ``MemoryStoreArchivedError`` rather than silently
    rewriting the row.  The service layer already guards this; the
    query layer is the defense-in-depth backstop for any future
    caller that bypasses the service."""
    pool, account_id, store_id = archived_memory_store

    async with pool.acquire() as conn:
        with pytest.raises(MemoryStoreArchivedError) as excinfo:
            await queries.update_memory_store(
                conn, store_id, name="post-archive", account_id=account_id
            )

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("id") == store_id

    # The row's name must not have been rewritten.
    async with pool.acquire() as conn:
        actual_name = await conn.fetchval("SELECT name FROM memory_stores WHERE id = $1", store_id)
    assert actual_name == "pre-archive"
