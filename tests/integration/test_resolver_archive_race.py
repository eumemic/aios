"""Integration test: the connector resolver's ``insert_chat_session``
paths must not race ``archive_connection`` into an
``archived AND active chat_session`` invariant violation.

``resolver._dispatch_routing_target`` and
``resolver._spawn_per_chat_session`` both insert a ``chat_sessions``
ledger row referencing the inbound's ``connection_id``. Pre-fix,
neither path acquired a ``SELECT … FOR UPDATE`` on ``connections``
before the insert.

The reachable race: an inbound arrives, the resolver loads the active
per_chat binding, then enters ``_spawn_per_chat_session`` —
operator concurrently detaches the binding and archives the
connection, both succeed (archive's active-binding check is satisfied
post-detach). Resolver's stale view of "binding active" continues to
the ``insert_chat_session`` call, which succeeds because the
``chat_sessions.connection_id`` FK accepts archived rows (no partial
``WHERE archived_at IS NULL``).

Symmetric to the ``bind_chat_to_session`` fix shipped in #663:
resolver takes the row lock + re-checks ``archived_at`` before the
insert, returning ``ResolveDrop.DETACHED`` if archived. Archive's
existing ``chat_sessions`` presence check (added in #663) closes the
opposite race direction.
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
from aios.services import agents as agents_service
from aios.services import connections as connections_service
from aios.services import environments as environments_service
from aios.services import session_templates as session_templates_service
from aios_connectors import resolver

pytestmark = pytest.mark.integration

_RACE_WINDOW_S = 0.2


@pytest.fixture
async def pool_with_per_chat_connection(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, connection_id)`` for acc_a with an active
    per_chat binding so the resolver's ``_spawn_per_chat_session``
    branch fires on resolve.
    """
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
        template = await session_templates_service.create_session_template(
            pool,
            account_id="acc_a",
            name="t",
            agent_id=agent_a.id,
            environment_id=env_a.id,
            agent_version=agent_a.version,
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
        )
        async with pool.acquire() as conn:
            connection = await queries.insert_connection(
                conn,
                account_id="acc_a",
                connector="signal",
                external_account_id="+15550001",
                metadata={},
            )
        await connections_service.configure_per_chat(
            pool,
            connection.id,
            account_id="acc_a",
            session_template_id=template.id,
        )
        yield pool, connection.id
    finally:
        await pool.close()


async def _count_chat_sessions(pool: asyncpg.Pool[Any], connection_id: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM chat_sessions WHERE connection_id = $1",
            connection_id,
        )


class TestResolverArchiveRace:
    async def test_resolve_during_concurrent_archive_preserves_invariant(
        self,
        pool_with_per_chat_connection: tuple[asyncpg.Pool[Any], str],
    ) -> None:
        """Resolver loads the active per_chat binding, then ``slow``
        ``get_session_template`` widens the window. Operator detaches +
        archives concurrently. Resolver's stale view continues to
        ``insert_chat_session``; without the resolver's FOR UPDATE +
        re-check, the chat_session lands on the now-archived connection
        — invariant violated.
        """
        pool, connection_id = pool_with_per_chat_connection

        original_get_session_template = queries.get_session_template

        async def slow_get_session_template(
            conn: asyncpg.Connection[Any], tid: str, *, account_id: str
        ) -> Any:
            result = await original_get_session_template(conn, tid, account_id=account_id)
            await asyncio.sleep(_RACE_WINDOW_S)
            return result

        async def _resolve() -> resolver.ResolveResult:
            async with pool.acquire() as conn:
                connection = await queries.get_connection(conn, connection_id, account_id="acc_a")
            return await resolver.resolve_target_session(
                pool,
                account_id="acc_a",
                connection=connection,
                chat_id="chat_x",
            )

        async def _detach_then_archive() -> None:
            await connections_service.unconfigure_connection(
                pool, connection_id, account_id="acc_a"
            )
            await connections_service.archive_connection(pool, connection_id, account_id="acc_a")

        with patch.object(queries, "get_session_template", slow_get_session_template):
            resolve_task = asyncio.create_task(_resolve())
            # Give resolver time to load the binding and enter
            # _spawn_per_chat_session, where it hits the slow
            # get_session_template call.
            await asyncio.sleep(_RACE_WINDOW_S / 4)
            await _detach_then_archive()
            await asyncio.wait_for(resolve_task, timeout=_RACE_WINDOW_S * 10)

        async with pool.acquire() as conn:
            connection = await queries.get_connection(conn, connection_id, account_id="acc_a")
        chat_count = await _count_chat_sessions(pool, connection_id)

        assert not (connection.archived_at is not None and chat_count > 0), (
            f"resolver+archive race produced "
            f"archived={connection.archived_at is not None} AND "
            f"chat_sessions={chat_count} — invariant violated."
        )
