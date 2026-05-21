"""``update_*`` query functions must close the archive race: a
concurrent ``archive_*`` committing between their upfront read and
their UPDATE must not result in a rewrite of the archived row.

This canary covers ``update_session``; the other update_* functions
in the family share the same UPDATE-WHERE fix, verified by the
sibling ``test_update_*_archived`` integration tests still passing.
The race is simulated deterministically via ``monkey-patch`` on
``get_session`` — feeding back the pre-archive snapshot while the
DB row is already archived bypasses the upfront check, which is
exactly the window the UPDATE WHERE clause has to close."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_session_with_stale_snapshot(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, Any]]:
    """Yield ``(pool, account_id, session_id, pre_archive_snapshot)``
    for a session that has been archived AFTER a snapshot was captured.

    The pre-archive snapshot is what ``get_session`` would return
    immediately before the racing archive committed.  Re-using it via
    monkey-patch lets the test deterministically reproduce the
    archive-race window without timing-dependent task interleaving.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_toctou', NULL, TRUE, 'toctou-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_toctou", prefix="toctou"
        )
        async with pool.acquire() as conn:
            snapshot = await queries.get_session(conn, session.id, account_id="acc_toctou")
            await queries.archive_session(conn, session.id, account_id="acc_toctou")
        assert snapshot.archived_at is None
        yield pool, "acc_toctou", session.id, snapshot
    finally:
        await pool.close()


async def test_update_session_closes_archive_toctou(
    archived_session_with_stale_snapshot: tuple[asyncpg.Pool[Any], str, str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, account_id, session_id, snapshot = archived_session_with_stale_snapshot

    async def fake_get_session(*_: Any, **__: Any) -> Any:
        return snapshot

    monkeypatch.setattr(queries, "get_session", fake_get_session)

    async with pool.acquire() as conn:
        with pytest.raises(ConflictError):
            await queries.update_session(
                conn,
                session_id,
                title="post-archive",
                account_id=account_id,
            )

    async with pool.acquire() as conn:
        actual_title = await conn.fetchval("SELECT title FROM sessions WHERE id = $1", session_id)
    assert actual_title == snapshot.title, (
        f"archived session row was rewritten despite the race: "
        f"title is {actual_title!r}, expected {snapshot.title!r}."
    )
