"""Integration tests for ``POST /v1/connectors/runtime/session-lifecycle`` (#1261).

The per-session-targeted, optionally-waking lifecycle route. Unlike the
broadcast ``/runtime/lifecycle`` route (which fans an event across every
session bound to a connection), this appends a single ``kind=lifecycle``
event onto **one** named session — the per-session gap the SMS design (§3.5
req 1) calls out: a delivery failure must reach the *originating* session, not
be broadcast.

Driven by calling the router handler ``post_runtime_session_lifecycle``
directly with a real pool and the resolved ``RuntimeAuthDep`` tuple (the
runtime-auth dependency resolves to a plain ``(token_id, connector,
account_id, connection_ids)`` tuple, so the handler is callable without an
HTTP layer). ``defer_wake`` is patched at the router's import site so the
wake assertion observes the call without a live procrastinate worker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.api.routers.connectors import (
    RuntimeSessionLifecycleRequest,
    post_runtime_session_lifecycle,
)
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ForbiddenError
from aios.services import agents as agents_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration

_CONNECTOR = "sms"


@pytest.fixture
async def pool_with_bound_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, connection_id, session_id)`` for an account
    whose session is bound (single_session) to an ``sms`` connection, plus an
    UNbound sibling session on the same account (for the 403 case)."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_sl', NULL, TRUE, 'tenant-sl')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_sl",
            name="sl-agent",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_sl", name="sl-env"
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_sl",
                agent_id=agent.id,
                environment_id=env.id,
                agent_version=agent.version,
                title=None,
                metadata={},
            )
            connection = await queries.insert_connection(
                conn,
                account_id="acc_sl",
                connector=_CONNECTOR,
                external_account_id="+15550001",
                metadata={},
            )
            await queries.insert_binding(
                conn,
                account_id="acc_sl",
                connection_id=connection.id,
                mode="single_session",
                session_id=session.id,
            )
        yield pool, "acc_sl", connection.id, session.id
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


class TestRuntimeSessionLifecycle:
    async def test_targets_exactly_the_one_session(
        self,
        pool_with_bound_session: tuple[asyncpg.Pool[Any], str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """The event lands on the named session only (not broadcast), with the
        broadcast-route payload shape."""
        pool, account_id, connection_id, session_id = pool_with_bound_session

        result = await post_runtime_session_lifecycle(
            RuntimeSessionLifecycleRequest(
                connection_id=connection_id,
                session_id=session_id,
                event="connector_delivery_failed",
                reason="30007",
                data={"detail": "carrier blocked", "peer": "+15550123"},
                wake=False,
            ),
            pool,
            _auth(account_id),
        )

        assert result["appended_session_ids"] == [session_id]
        assert result["woke"] is False
        rows = await _lifecycle_rows(pool, session_id)
        assert len(rows) == 1
        payload = rows[0]
        assert payload["event"] == "connector_delivery_failed"
        assert payload["connection_id"] == connection_id
        assert payload["connector"] == _CONNECTOR
        assert payload["reason"] == "30007"
        assert payload["data"] == {"detail": "carrier blocked", "peer": "+15550123"}

    async def test_wake_true_enqueues_wake_false_does_not(
        self,
        pool_with_bound_session: tuple[asyncpg.Pool[Any], str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        pool, account_id, connection_id, session_id = pool_with_bound_session

        await post_runtime_session_lifecycle(
            RuntimeSessionLifecycleRequest(
                connection_id=connection_id,
                session_id=session_id,
                event="connector_delivery_failed",
                wake=False,
            ),
            pool,
            _auth(account_id),
        )
        assert patched_defer_wake.await_count == 0

        await post_runtime_session_lifecycle(
            RuntimeSessionLifecycleRequest(
                connection_id=connection_id,
                session_id=session_id,
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

    async def test_forbidden_when_session_not_bound(
        self,
        pool_with_bound_session: tuple[asyncpg.Pool[Any], str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """An unbound (but same-account) session is rejected — a runtime bearer
        can only target a session genuinely bound to one of its connections."""
        pool, account_id, connection_id, _session_id = pool_with_bound_session
        # A fresh session on the same account, NOT bound to the connection.
        from tests.integration.conftest import seed_agent_env_session

        _agent, _env, unbound = await seed_agent_env_session(
            pool, account_id=account_id, prefix="sl-unbound"
        )

        with pytest.raises(ForbiddenError):
            await post_runtime_session_lifecycle(
                RuntimeSessionLifecycleRequest(
                    connection_id=connection_id,
                    session_id=unbound.id,
                    event="connector_delivery_failed",
                    wake=True,
                ),
                pool,
                _auth(account_id),
            )
        assert patched_defer_wake.await_count == 0
        assert await _lifecycle_rows(pool, unbound.id) == []

    async def test_forbidden_when_bearer_connector_mismatch(
        self,
        pool_with_bound_session: tuple[asyncpg.Pool[Any], str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """A bearer scoped to a different connector type cannot reach this
        connection (``_check_runtime_scope``)."""
        pool, account_id, connection_id, session_id = pool_with_bound_session

        with pytest.raises(ForbiddenError):
            await post_runtime_session_lifecycle(
                RuntimeSessionLifecycleRequest(
                    connection_id=connection_id,
                    session_id=session_id,
                    event="connector_delivery_failed",
                ),
                pool,
                _auth(account_id, connector="whatsapp"),
            )
        assert patched_defer_wake.await_count == 0


class TestRuntimeSessionLifecycleDeliveryAck:
    """The success-path delivered/edited acks (#1341) ride the same route as
    ``connector_delivery_failed``: append a targeted ``kind=lifecycle`` row
    with the reserved ``event`` and a ``platform_message_id``/``tool_call_id``
    ``data`` payload, ``wake=False`` by default (informational)."""

    async def test_session_lifecycle_delivered_ack_visible(
        self,
        pool_with_bound_session: tuple[asyncpg.Pool[Any], str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        pool, account_id, connection_id, session_id = pool_with_bound_session

        result = await post_runtime_session_lifecycle(
            RuntimeSessionLifecycleRequest(
                connection_id=connection_id,
                session_id=session_id,
                event="connector_message_delivered",
                data={"platform_message_id": "x", "tool_call_id": "call_1"},
                wake=False,
            ),
            pool,
            _auth(account_id),
        )

        assert result["appended_session_ids"] == [session_id]
        assert result["woke"] is False
        assert patched_defer_wake.await_count == 0
        rows = await _lifecycle_rows(pool, session_id)
        assert len(rows) == 1
        payload = rows[0]
        assert payload["event"] == "connector_message_delivered"
        assert payload["connection_id"] == connection_id
        assert payload["data"] == {"platform_message_id": "x", "tool_call_id": "call_1"}

        # The windowed read (which feeds build_messages) admits the ack row.
        from aios.db.queries.events import read_windowed_context_events

        async with pool.acquire() as conn:
            windowed = await read_windowed_context_events(
                conn, session_id, account_id=account_id
            )
        acks = [e for e in windowed if e.kind == "lifecycle"]
        assert len(acks) == 1
        assert acks[0].data["event"] == "connector_message_delivered"

    async def test_session_lifecycle_ack_wake_defers(
        self,
        pool_with_bound_session: tuple[asyncpg.Pool[Any], str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """``wake=True`` defers a wake exactly once (the route's path, not the
        renderer's) — acks default to ``wake=False`` but the param still works."""
        pool, account_id, connection_id, session_id = pool_with_bound_session

        await post_runtime_session_lifecycle(
            RuntimeSessionLifecycleRequest(
                connection_id=connection_id,
                session_id=session_id,
                event="connector_message_delivered",
                data={"platform_message_id": "x"},
                wake=True,
            ),
            pool,
            _auth(account_id),
        )
        assert patched_defer_wake.await_count == 1

    async def test_session_lifecycle_ack_unbound_403(
        self,
        pool_with_bound_session: tuple[asyncpg.Pool[Any], str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """An ack targeting a session not bound to the connection is rejected
        by the shipped bound-session gate."""
        pool, account_id, connection_id, _session_id = pool_with_bound_session
        from tests.integration.conftest import seed_agent_env_session

        _agent, _env, unbound = await seed_agent_env_session(
            pool, account_id=account_id, prefix="sl-ack-unbound"
        )

        with pytest.raises(ForbiddenError):
            await post_runtime_session_lifecycle(
                RuntimeSessionLifecycleRequest(
                    connection_id=connection_id,
                    session_id=unbound.id,
                    event="connector_message_delivered",
                    data={"platform_message_id": "x"},
                ),
                pool,
                _auth(account_id),
            )
        assert await _lifecycle_rows(pool, unbound.id) == []

    async def test_windowed_read_admits_ack_drops_unknown(
        self,
        pool_with_bound_session: tuple[asyncpg.Pool[Any], str, str, str],
        patched_defer_wake: AsyncMock,
    ) -> None:
        """The windowed read admits the allowlisted ack but drops a
        non-allowlisted ``event`` (the load-bearing allowlist edit)."""
        from aios.db.queries.events import read_windowed_context_events
        from aios.services import sessions as sessions_service

        pool, account_id, _connection_id, session_id = pool_with_bound_session

        await sessions_service.append_event(
            pool,
            session_id,
            "lifecycle",
            {"event": "connector_message_delivered", "data": {"platform_message_id": "x"}},
            account_id=account_id,
        )
        await sessions_service.append_event(
            pool,
            session_id,
            "lifecycle",
            {"event": "some_unknown_connector_event"},
            account_id=account_id,
        )

        async with pool.acquire() as conn:
            windowed = await read_windowed_context_events(
                conn, session_id, account_id=account_id
            )
        lifecycle_events = [e.data["event"] for e in windowed if e.kind == "lifecycle"]
        assert "connector_message_delivered" in lifecycle_events
        assert "some_unknown_connector_event" not in lifecycle_events
