"""Integration test: ``bind_chat_to_session`` and ``archive_connection``
must not race into an invariant-violating end state (archived
connection row + active ``chat_sessions`` row pointing at it).

Pre-fix, ``bind_chat_to_session`` validated the connection via a
plain ``get_connection`` read inside its transaction — no
``SELECT FOR UPDATE`` row lock on ``connections``. A concurrent
``archive_connection`` could commit between the read and the
``insert_chat_session`` write. The ``chat_sessions.connection_id``
FK accepts archived rows (no partial ``WHERE archived_at IS NULL``),
so the insert succeeded and the invariant violation persisted until
the resolver's tier-1 DETACH safety net (#526/#541) fired on the
next inbound.

Two race directions, both pinned here:

* **archive-first**: slow ``get_active_binding`` widens archive's
  window. Archive holds ``FOR UPDATE`` on the connections row
  through the sleep; bind blocks behind the lock until archive
  commits, then re-reads ``archived_at`` and raises ``ConflictError``.

* **bind-first**: slow ``get_session`` widens bind's window. Bind
  holds ``FOR UPDATE`` through the sleep; archive blocks until bind
  commits the chat_session. Archive then sees the chat_session via
  its added presence check and raises ``ConflictError``.

The fix is symmetric: bind takes the row lock + re-checks
``archived_at``; archive's existing under-lock check is extended to
include ``chat_sessions`` so it refuses to archive while any
operator-curated chat row references the connection.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.services import agents as agents_service
from aios.services import connections as connections_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration

_RACE_WINDOW_S = 0.2


@pytest.fixture
async def pool_with_connection_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, connection_id, session_id)`` for acc_a."""
    pool = await create_pool(migrated_db_url, min_size=2, max_size=8)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root', NULL,       TRUE,  'tenant-root'),
                       ('acc_a',    'acc_root', FALSE, 'tenant-a')
                """
            )

        agent_a = await agents_service.create_agent(
            pool,
            account_id="acc_a",
            name="a-agent",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env_a = await environments_service.create_environment(
            pool, account_id="acc_a", name="a-env"
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_a",
                agent_id=agent_a.id,
                environment_id=env_a.id,
                agent_version=agent_a.version,
                title=None,
                metadata={},
            )
            connection = await queries.insert_connection(
                conn,
                account_id="acc_a",
                connector="signal",
                external_account_id="+15550001",
                metadata={},
            )
        yield pool, connection.id, session.id
    finally:
        await pool.close()


async def _count_active_chat_sessions(pool: asyncpg.Pool[Any], connection_id: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM chat_sessions WHERE connection_id = $1",
            connection_id,
        )


class TestBindChatArchiveRace:
    async def test_archive_first_then_bind_raises(
        self,
        pool_with_connection_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """Archive holds ``FOR UPDATE`` (widened via slow
        ``get_active_binding``); concurrent bind blocks then re-reads
        ``archived_at`` under the lock and raises ``ConflictError``.
        End state: archived, NO chat_sessions."""
        pool, connection_id, session_id = pool_with_connection_and_session

        original = queries.get_active_binding

        async def slow_get_active_binding(
            conn: asyncpg.Connection[Any], cid: str, *, account_id: str
        ) -> Any:
            result = await original(conn, cid, account_id=account_id)
            await asyncio.sleep(_RACE_WINDOW_S)
            return result

        archive_error: BaseException | None = None
        bind_error: BaseException | None = None

        async def _archive() -> None:
            nonlocal archive_error
            try:
                await connections_service.archive_connection(
                    pool, connection_id, account_id="acc_a"
                )
            except BaseException as exc:
                archive_error = exc

        async def _bind() -> None:
            nonlocal bind_error
            try:
                await connections_service.bind_chat_to_session(
                    pool,
                    connection_id,
                    account_id="acc_a",
                    chat_id="chat_x",
                    session_id=session_id,
                )
            except BaseException as exc:
                bind_error = exc

        with patch.object(queries, "get_active_binding", slow_get_active_binding):
            archive_task = asyncio.create_task(_archive())
            await asyncio.sleep(_RACE_WINDOW_S / 4)
            bind_task = asyncio.create_task(_bind())
            await asyncio.wait_for(
                asyncio.gather(archive_task, bind_task, return_exceptions=True),
                timeout=_RACE_WINDOW_S * 10,
            )

        async with pool.acquire() as conn:
            connection = await queries.get_connection(conn, connection_id, account_id="acc_a")
        chat_count = await _count_active_chat_sessions(pool, connection_id)

        assert not (connection.archived_at is not None and chat_count > 0), (
            f"archive-first race produced archived={connection.archived_at is not None} "
            f"AND chat_sessions={chat_count} — invariant violated. "
            f"archive_error={archive_error!r} bind_error={bind_error!r}"
        )
        # Bind should have raised (archive won the lock).
        assert isinstance(bind_error, ConflictError), (
            f"bind should have raised ConflictError after archive committed; got {bind_error!r}"
        )

    async def test_bind_first_then_archive_raises(
        self,
        pool_with_connection_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """Bind holds ``FOR UPDATE`` (widened via slow ``get_session``);
        concurrent archive blocks then sees the inserted chat_session
        via its added presence check and raises ``ConflictError``.
        End state: NOT archived, chat_sessions row exists."""
        pool, connection_id, session_id = pool_with_connection_and_session

        original_get_session = queries.get_session

        async def slow_get_session(
            conn: asyncpg.Connection[Any], sid: str, *, account_id: str
        ) -> Any:
            result = await original_get_session(conn, sid, account_id=account_id)
            await asyncio.sleep(_RACE_WINDOW_S)
            return result

        archive_error: BaseException | None = None
        bind_error: BaseException | None = None

        async def _archive() -> None:
            nonlocal archive_error
            try:
                await connections_service.archive_connection(
                    pool, connection_id, account_id="acc_a"
                )
            except BaseException as exc:
                archive_error = exc

        async def _bind() -> None:
            nonlocal bind_error
            try:
                await connections_service.bind_chat_to_session(
                    pool,
                    connection_id,
                    account_id="acc_a",
                    chat_id="chat_y",
                    session_id=session_id,
                )
            except BaseException as exc:
                bind_error = exc

        with patch.object(queries, "get_session", slow_get_session):
            bind_task = asyncio.create_task(_bind())
            await asyncio.sleep(_RACE_WINDOW_S / 4)
            archive_task = asyncio.create_task(_archive())
            await asyncio.wait_for(
                asyncio.gather(archive_task, bind_task, return_exceptions=True),
                timeout=_RACE_WINDOW_S * 10,
            )

        async with pool.acquire() as conn:
            connection = await queries.get_connection(conn, connection_id, account_id="acc_a")
        chat_count = await _count_active_chat_sessions(pool, connection_id)

        assert not (connection.archived_at is not None and chat_count > 0), (
            f"bind-first race produced archived={connection.archived_at is not None} "
            f"AND chat_sessions={chat_count} — invariant violated. "
            f"archive_error={archive_error!r} bind_error={bind_error!r}"
        )
        # Archive should have raised (bind won the lock; chat_sessions
        # exists now, archive's chat_sessions presence check refuses).
        assert isinstance(archive_error, ConflictError), (
            f"archive should have raised ConflictError after bind committed "
            f"a chat_session; got {archive_error!r}"
        )
