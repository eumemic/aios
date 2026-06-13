"""Integration tests: ``archive_session`` fires a NOTIFY so mid-flight
session listeners wake (issue #906).

``db/queries/sessions.archive_session`` only flips ``archived_at`` — it appends
no event, so before this fix any consumer blocked on ``events_<session_id>``
(the ``await`` primitive, the long-poll ``/wait`` endpoint, the SSE ``/stream``)
got no wake signal on mid-flight archival and sat until its own timeout.

The service-layer ``archive_session`` now fires a bare
``EVENTS_ARCHIVED_NOTIFY`` poke on ``events_<session_id>`` AFTER the archive
transaction commits (the NOTIFY-after-commit invariant). These tests verify:

* the raw poke lands on the channel after commit, and
* a Mode-1 (request_id) ``await_session`` blocked mid-flight wakes on the poke
  and returns ``child_gone`` promptly — well before its own timeout.

DB-backed (testcontainer Postgres).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.listen import EVENTS_ARCHIVED_NOTIFY, open_listen_for_events
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.services import sessions as service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a fresh session."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_arch_notify', NULL, TRUE, 'archive-notify-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_arch_notify", prefix="archive-notify-test"
        )
        yield pool, "acc_arch_notify", session.id
    finally:
        await pool.close()


async def test_archive_fires_notify_on_events_channel(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    """A subscriber LISTENing on ``events_<id>`` receives the archive sentinel
    after the archive commits."""
    pool, account_id, session_id = pool_and_session

    subscription = await open_listen_for_events(migrated_db_url, session_id, acquire_lock=False)
    try:
        await service.archive_session(pool, session_id, account_id=account_id)
        payload = await asyncio.wait_for(subscription.queue.get(), timeout=5)
        assert payload == EVENTS_ARCHIVED_NOTIFY
    finally:
        subscription.terminate()


async def test_mode1_await_wakes_promptly_on_mid_flight_archive(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    """A Mode-1 await blocked on an unresolved request wakes on the archive
    poke and returns ``child_gone`` — without sitting out its full timeout."""
    pool, account_id, session_id = pool_and_session

    async def archive_late() -> None:
        await asyncio.sleep(0.2)
        await service.archive_session(pool, session_id, account_id=account_id)

    loop = asyncio.get_running_loop()
    start = loop.time()
    resp, _ = await asyncio.gather(
        service.await_session(
            pool,
            migrated_db_url,
            session_id,
            account_id=account_id,
            request_id="req_archived",
            watermark=None,
            timeout_seconds=30,
        ),
        archive_late(),
    )
    elapsed = loop.time() - start

    assert resp.done is True
    assert resp.is_error is True
    assert resp.error == {"kind": "child_gone"}
    # The poke woke the await; it must not have waited out the 30s timeout.
    assert elapsed < 10


async def test_archive_already_archived_does_not_fire_second_poke(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    """Re-archiving an already-archived session raises NotFound (the query is
    fenced by ``archived_at IS NULL``) BEFORE the poke — so no spurious poke
    fires for a no-op archive."""
    pool, account_id, session_id = pool_and_session
    await service.archive_session(pool, session_id, account_id=account_id)

    subscription = await open_listen_for_events(migrated_db_url, session_id, acquire_lock=False)
    try:
        with pytest.raises(NotFoundError):
            await service.archive_session(pool, session_id, account_id=account_id)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(subscription.queue.get(), timeout=0.5)
    finally:
        subscription.terminate()
