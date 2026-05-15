"""Integration test: ``list_memory_versions`` orders deterministically when
``created_at`` ties.

Pre-fix: ``ORDER BY created_at DESC`` was the only sort key
(``queries.py:4671``). Postgres default ``now()`` resolves to the
transaction-start time, so multiple ``memory_versions`` rows written in
the same transaction share ``created_at`` to the microsecond. Without a
tiebreaker, page output for tied rows is unspecified — Postgres may
return them in any order, and that order can shift across replan / row
GC / table rewrite. Clients paging through the version history of a
bulk-edited store would observe non-deterministic page boundaries.

Bulk-edits are routine: the standard ``update_memory_with_version``
flow at ``queries.py:4540`` runs inside ``conn.transaction()``, and an
operator who edits N memories in one HTTP request (or any caller that
batches writes) gets N versions with shared ``created_at``.

Fix: add ``seq DESC`` as a tiebreaker. The ``memory_versions`` table
has a ``UNIQUE (memory_store_id, seq)`` constraint allocated monotonic-
ally by ``_allocate_version_seq`` — so within a store, ``seq DESC`` is
unambiguous AND preserves the "newest first" intent (higher seq =
later in this txn) even when ``created_at`` is tied.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.ids import make_id

pytestmark = pytest.mark.integration


@pytest.fixture
async def store_with_seeded_memory(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, store_id, memory_id)`` for a seeded
    memory_store + memory ready to receive version rows."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_test', NULL, TRUE, 'tenant-test')
                """
            )
            store_id = make_id("memstore")
            await conn.execute(
                """
                INSERT INTO memory_stores (id, name, account_id)
                VALUES ($1, 'test-store', 'acc_test')
                """,
                store_id,
            )
            memory_id = make_id("mem")
            await conn.execute(
                """
                INSERT INTO memories (
                    id, memory_store_id, path, content, content_sha256,
                    content_size_bytes, account_id
                ) VALUES ($1, $2, '/a.txt', 'hi', 'sha', 2, 'acc_test')
                """,
                memory_id,
                store_id,
            )
        yield pool, "acc_test", store_id, memory_id
    finally:
        await pool.close()


async def _insert_version_at(
    pool: asyncpg.Pool[Any],
    *,
    store_id: str,
    memory_id: str,
    account_id: str,
    seq: int,
    created_at: datetime,
) -> str:
    """Insert one memory_versions row with a controlled created_at + seq."""
    version_id = make_id("memver")
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_versions (
                id, memory_store_id, memory_id, seq, operation,
                path, content, content_sha256, content_size_bytes,
                created_by_type, created_by_ref, created_at, account_id
            )
            VALUES ($1, $2, $3, $4, 'modified',
                    '/a.txt', 'hi', 'sha', 2,
                    'api_actor', 'test', $5, $6)
            """,
            version_id,
            store_id,
            memory_id,
            seq,
            created_at,
            account_id,
        )
    return version_id


async def test_list_memory_versions_is_deterministic_when_created_at_ties(
    store_with_seeded_memory: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """When multiple versions share ``created_at``, the listing must be
    deterministic and ordered by ``seq DESC`` (newest first).

    Pre-fix the ORDER BY had only ``created_at DESC``; the tied rows
    came back in Postgres' internal order, which is not seq-aligned for
    a non-monotonic insertion sequence.
    """
    pool, account_id, store_id, memory_id = store_with_seeded_memory

    tied_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Insert seq=2, then seq=1, then seq=3 — deliberately non-monotonic
    # insertion order so Postgres' internal heap order differs from
    # ``seq DESC``. (If the planner happens to scan in any order other
    # than DESC, the pre-fix assertion will fail.)
    id_for_seq: dict[int, str] = {}
    for seq in [2, 1, 3]:
        id_for_seq[seq] = await _insert_version_at(
            pool,
            store_id=store_id,
            memory_id=memory_id,
            account_id=account_id,
            seq=seq,
            created_at=tied_ts,
        )

    async with pool.acquire() as conn:
        versions = await queries.list_memory_versions(
            conn, store_id, account_id=account_id, memory_id=memory_id
        )

    returned_ids = [v.id for v in versions]
    expected_ids = [id_for_seq[3], id_for_seq[2], id_for_seq[1]]
    assert returned_ids == expected_ids, (
        f"versions with tied created_at returned in order {returned_ids!r}; "
        f"expected {expected_ids!r} (seq DESC: 3, 2, 1). Without a "
        f"seq-tiebreaker the ORDER BY relies on Postgres' internal heap "
        f"order, which is unspecified — bulk-write flows leave tied "
        f"created_at across all rows of one transaction."
    )


async def test_list_memory_versions_distinct_timestamps_still_works(
    store_with_seeded_memory: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """Regression guard: when ``created_at`` values differ, the
    primary sort by created_at DESC still wins (the seq tiebreaker only
    activates on ties)."""
    pool, account_id, store_id, memory_id = store_with_seeded_memory

    # Earlier timestamp gets higher seq — tests that created_at DESC
    # still beats seq DESC for the primary sort.
    older_id = await _insert_version_at(
        pool,
        store_id=store_id,
        memory_id=memory_id,
        account_id=account_id,
        seq=5,
        created_at=datetime(2026, 1, 1, 11, 0, 0, tzinfo=UTC),
    )
    newer_id = await _insert_version_at(
        pool,
        store_id=store_id,
        memory_id=memory_id,
        account_id=account_id,
        seq=1,
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
    )

    async with pool.acquire() as conn:
        versions = await queries.list_memory_versions(
            conn, store_id, account_id=account_id, memory_id=memory_id
        )

    returned_ids = [v.id for v in versions]
    assert returned_ids == [newer_id, older_id], (
        f"expected newer-by-created_at first: [{newer_id!r}, {older_id!r}]; got {returned_ids!r}"
    )
