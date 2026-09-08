"""``archive_child`` must close the archive-TOCTOU race: a concurrent
``purge_account`` hard-deleting the (already-archived) child between
``archive_child``'s scope check and its archive ``UPDATE`` must surface
as a clean ``ConflictError`` (409), not an uncaught ``AssertionError``
that Starlette turns into an HTTP 500 with no aios error envelope.

The scope check (``get_account_in_scope``) already raised
``NotFoundError`` on a missing row, so a no-row return from
``archive_account`` *after* the scope check means the row vanished
mid-call — exactly what ``purge_account``'s hard delete produces. This
mirrors the established codebase precedent for the same archive-TOCTOU
class (the ``update_*`` family's no-row-after-race branch maps to
``ConflictError``): the no-row post-UPDATE is the TOCTOU losing the
race.

A deterministic canary reproduces the post-race state (scope check sees
a stale "row exists" snapshot via monkey-patch while the DB row is
already hard-deleted → ``archive_account`` returns ``None``) without
timing-dependent task interleaving — the honest artifact for this
class. Plus an idempotent re-archive guard so the fix cannot over-broadly
map the already-archived path (which returns the row unchanged) to a 409.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.models.accounts import Account
from aios.services import accounts as service

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_child_with_stale_snapshot(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, Account]]:
    """Yield ``(pool, caller_id, target_id, pre_delete_snapshot)`` for an
    already-archived child whose row has been hard-deleted (as a concurrent
    ``purge_account`` would) AFTER a snapshot was captured.

    The pre-delete snapshot is what ``get_account`` would have returned
    immediately before the racing purge committed its ``DELETE``. Re-using
    it via monkey-patch reproduces the archive-race window deterministically
    — no timing-dependent task interleaving: the scope check sees the stale
    "row exists" snapshot while the DB row is already gone, so
    ``archive_account`` returns ``None``, the exact post-race state the
    TOCTOU branch must handle.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_parent', NULL,        TRUE,  'parent'),
                       ('acc_child',  'acc_parent', FALSE, 'child')
                """
            )
            await queries.archive_account(conn, "acc_child")
        async with pool.acquire() as conn:
            snapshot = await queries.get_account(conn, "acc_child")
        assert snapshot is not None, "child row should exist after archive"
        assert snapshot.archived_at is not None, "child should be archived"
        # Simulate the concurrent purge hard-deleting the archived child.
        async with pool.acquire() as conn:
            deleted = await queries.hard_delete_account(conn, "acc_child")
        assert deleted, "hard-delete of an archived childless account should succeed"
        yield pool, "acc_parent", "acc_child", snapshot
    finally:
        await pool.close()


async def test_archive_child_raises_conflict_when_purge_wins_race(
    archived_child_with_stale_snapshot: tuple[asyncpg.Pool[Any], str, str, Account],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOCTOU-loss branch: scope check passed (stale snapshot), but the row is
    gone by the UPDATE. Must raise ``ConflictError``, not the pre-fix
    ``AssertionError`` that would surface as an unenveloped HTTP 500.
    """
    pool, caller_id, target_id, snapshot = archived_child_with_stale_snapshot

    async def fake_get_account(*_: Any, **__: Any) -> Account:
        return snapshot

    # The scope check calls ``queries.get_account``; feeding the pre-delete
    # snapshot makes it pass while the real row is already gone.
    # ``archive_account``'s re-read is raw SQL (not ``queries.get_account``),
    # so it correctly observes the row is missing → returns None.
    monkeypatch.setattr(queries, "get_account", fake_get_account)

    with pytest.raises(ConflictError):
        await service.archive_child(pool, target_account_id=target_id, caller_account_id=caller_id)

    # No resurrection: the losing archive wrote nothing; the row stays gone.
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM accounts WHERE id = $1", target_id)
    assert row is None, "concurrent-purge row should not be re-created by the losing archive"


async def test_archive_child_idempotent_rearchive_returns_row(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """Re-archiving an already-archived child is idempotent: returns the
    archived row (with its original ``archived_at``), NOT a ``ConflictError``.

    Guards the fix from over-broadly mapping the already-archived path to a
    409 — that path returns the row unchanged, so ``archived`` is not None.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_parent', NULL,        TRUE,  'parent'),
                       ('acc_child',  'acc_parent', FALSE, 'child')
                """
            )
        first = await service.archive_child(
            pool, target_account_id="acc_child", caller_account_id="acc_parent"
        )
        assert first.archived_at is not None
        second = await service.archive_child(
            pool, target_account_id="acc_child", caller_account_id="acc_parent"
        )
        assert second.archived_at is not None
        assert second.archived_at == first.archived_at, (
            "idempotent re-archive must not re-stamp archived_at"
        )
    finally:
        await pool.close()
