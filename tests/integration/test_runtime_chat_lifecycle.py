"""Integration tests for ``POST /v1/connectors/runtime/chat-lifecycle`` (#1260).

The routing-key (``chat_id``) variant of the session-targeted lifecycle
append. Where ``/runtime/session-lifecycle`` (#1261) needs the caller to
already hold the resolved ``session_id``, this route carries the connector's
per-peer routing key and resolves it through the connection's per-chat
binding server-side — the SMS design's §3.5 req 1 second option ("route the
per-peer failure through the resolver on the callback's ``To``").

Like the session-lifecycle suite, the route handler ``post_runtime_chat_lifecycle``
is driven directly with a real pool and a resolved ``RuntimeAuthDep`` tuple;
``defer_wake`` is patched at the router's import site so the wake assertion
observes the call without a live procrastinate worker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.api.routers.connectors import (
    RuntimeChatLifecycleRequest,
    post_runtime_chat_lifecycle,
)
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ForbiddenError, NotFoundError
from aios.services import agents as agents_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration

_CONNECTOR = "sms"
_CHAT_ID = "+15550123"


@pytest.fixture
async def pool_with_chat_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str, str]]:
    """Yield ``(pool, account_id, connection_id, chat_id, session_id)`` for an
    account whose per-chat session (``chat_sessions`` row keyed on ``chat_id``)
    is bound to an ``sms`` connection."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_cl', NULL, TRUE, 'tenant-cl')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_cl",
            name="cl-agent",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_cl", name="cl-env"
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_cl",
                agent_id=agent.id,
                environment_id=env.id,
                agent_version=agent.version,
                title=None,
                metadata={},
            )
            connection = await queries.insert_connection(
                conn,
                account_id="acc_cl",
                connector=_CONNECTOR,
                external_account_id="+15550001",
                metadata={},
            )
            # Per-chat binding: the routing key (chat_id) → session the
            # resolver will return. No single_session binding here — this is
            # exactly the per_chat shape the SMS connector spawns per peer.
            await queries.insert_chat_session(
                conn,
                account_id="acc_cl",
                connection_id=connection.id,
                chat_id=_CHAT_ID,
                session_id=session.id,
            )
        yield pool, "acc_cl", connection.id, _CHAT_ID, session.id
    finally:
        await pool.close()


@pytest.fixture
def patched_defer_wake() -> Any:
    """Patch the router's ``defer_wake`` import site so the wake enqueue is a
    no-op the test can assert against (no live procrastinate worker needed)."""
    with mock.patch("aios.api.routers.connectors.defer_wake", new_callable=AsyncMock) as m:
        yield m


def _auth(account_id: str, connector: str = _CONNECTOR) -> tuple[str, str, str, None]:
    """Shape of a resolved unscoped ``RuntimeAuthDep`` tuple:
    ``(token_id, connector, account_id, connection_ids)``."""
    return ("rt_test", connector, account_id, None)


async def _lifecycle_rows(pool: asyncpg.Pool[Any], session_id: str) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'lifecycle' ORDER BY seq",
            session_id,
        )
    import json

    return [json.loads(r["data"]) if isinstance(r["data"], str) else r["data"] for r in rows]


class TestRuntimeChatLifecycle:
    async def test_resolves_chat_id_to_the_one_session(
        self,
        pool_with_chat_session: tuple[asyncpg.Pool[Any], str, str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """The event lands on the session the chat_id resolves to (not
        broadcast), carrying the chat_id in the payload alongside the
        broadcast-route fields."""
        pool, account_id, connection_id, chat_id, session_id = pool_with_chat_session

        result = await post_runtime_chat_lifecycle(
            RuntimeChatLifecycleRequest(
                connection_id=connection_id,
                chat_id=chat_id,
                event="connector_delivery_failed",
                reason="30007",
                data={"detail": "carrier blocked", "peer": chat_id},
                wake=False,
            ),
            pool,
            _auth(account_id),
        )

        assert result["appended_session_ids"] == [session_id]
        assert result["session_id"] == session_id
        assert result["woke"] is False
        rows = await _lifecycle_rows(pool, session_id)
        assert len(rows) == 1
        payload = rows[0]
        assert payload["event"] == "connector_delivery_failed"
        assert payload["connection_id"] == connection_id
        assert payload["connector"] == _CONNECTOR
        assert payload["chat_id"] == chat_id
        assert payload["reason"] == "30007"
        assert payload["data"] == {"detail": "carrier blocked", "peer": chat_id}

    async def test_wake_true_enqueues_wake_false_does_not(
        self,
        pool_with_chat_session: tuple[asyncpg.Pool[Any], str, str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        pool, account_id, connection_id, chat_id, session_id = pool_with_chat_session

        await post_runtime_chat_lifecycle(
            RuntimeChatLifecycleRequest(
                connection_id=connection_id,
                chat_id=chat_id,
                event="connector_delivery_failed",
                wake=False,
            ),
            pool,
            _auth(account_id),
        )
        assert patched_defer_wake.await_count == 0

        await post_runtime_chat_lifecycle(
            RuntimeChatLifecycleRequest(
                connection_id=connection_id,
                chat_id=chat_id,
                event="connector_delivery_failed",
                wake=True,
            ),
            pool,
            _auth(account_id),
        )
        assert patched_defer_wake.await_count == 1
        assert patched_defer_wake.await_args is not None
        args = patched_defer_wake.await_args.args
        kwargs = patched_defer_wake.await_args.kwargs
        assert args[1] == session_id
        assert kwargs["account_id"] == account_id
        assert kwargs["cause"] == "connector_lifecycle"

    async def test_not_found_when_chat_id_unbound(
        self,
        pool_with_chat_session: tuple[asyncpg.Pool[Any], str, str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """An unknown routing key resolves to no session — 404 (drop rather
        than fan a spurious cross-peer notice), with no append and no wake."""
        pool, account_id, connection_id, _chat_id, _session_id = pool_with_chat_session

        with pytest.raises(NotFoundError):
            await post_runtime_chat_lifecycle(
                RuntimeChatLifecycleRequest(
                    connection_id=connection_id,
                    chat_id="+19999999",
                    event="connector_delivery_failed",
                    wake=True,
                ),
                pool,
                _auth(account_id),
            )
        assert patched_defer_wake.await_count == 0

    async def test_forbidden_when_bearer_connector_mismatch(
        self,
        pool_with_chat_session: tuple[asyncpg.Pool[Any], str, str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """A bearer scoped to a different connector type cannot reach this
        connection (``_check_runtime_scope``) — even before resolution."""
        pool, account_id, connection_id, chat_id, _session_id = pool_with_chat_session

        with pytest.raises(ForbiddenError):
            await post_runtime_chat_lifecycle(
                RuntimeChatLifecycleRequest(
                    connection_id=connection_id,
                    chat_id=chat_id,
                    event="connector_delivery_failed",
                ),
                pool,
                _auth(account_id, connector="whatsapp"),
            )
        assert patched_defer_wake.await_count == 0
