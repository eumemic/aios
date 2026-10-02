"""Integration test: ``POST /v1/connectors/signal/unregister`` must refuse
(409, no management call dispatched) while the number belongs to a LIVE
connection — owned by the caller, not archived, still bound to a session.

Unregistering is irreversible, so the documented "detach (or archive)
first" precondition is enforced server-side against real connection +
binding state, not a stub.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from aios.api.routers.connectors import SignalUnregisterRequest, post_signal_unregister
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.services import connections as connections_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

PHONE = "+16575274288"


@pytest.fixture
async def attached_signal(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, connection_id, session_id)`` — a signal connection
    for ``PHONE`` owned by ``acc_owner`` and attached to a session; a second
    tenant ``acc_other`` exists with no connections."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root',  NULL,       TRUE,  'root'),
                       ('acc_owner', 'acc_root', FALSE, 'owner'),
                       ('acc_other', 'acc_root', FALSE, 'other')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_owner", prefix="sig-unreg"
        )
        async with pool.acquire() as conn:
            connection = await queries.insert_connection(
                conn,
                connector="signal",
                external_account_id=PHONE,
                metadata={},
                account_id="acc_owner",
            )
        await connections_service.attach_connection(
            pool, connection.id, account_id="acc_owner", session_id=session.id
        )
        yield pool, connection.id, session.id
    finally:
        await pool.close()


async def _unregister(pool: asyncpg.Pool[Any], phone: str, account_id: str) -> AsyncMock:
    submit = AsyncMock(return_value=({"external_account_id": phone}, False))
    with patch("aios.api.routers.connectors.management_calls.submit_call", submit):
        await post_signal_unregister(
            SignalUnregisterRequest(external_account_id=phone), "postgresql://x", pool, account_id
        )
    return submit


async def test_refused_while_attached_even_with_formatting_differences(
    attached_signal: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, connection_id, _session_id = attached_signal
    for phone in (PHONE, "1 (657) 527-4288", "16575274288"):
        submit = AsyncMock(return_value=({}, False))
        with (
            patch("aios.api.routers.connectors.management_calls.submit_call", submit),
            pytest.raises(ConflictError) as exc_info,
        ):
            await post_signal_unregister(
                SignalUnregisterRequest(external_account_id=phone),
                "postgresql://x",
                pool,
                "acc_owner",
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["connection_ids"] == [connection_id]
        assert submit.await_count == 0


async def test_allowed_after_detach(
    attached_signal: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, connection_id, _ = attached_signal
    await connections_service.detach_connection(pool, connection_id, account_id="acc_owner")
    submit = await _unregister(pool, PHONE, "acc_owner")
    assert submit.await_count == 1


async def test_allowed_after_archive(
    attached_signal: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, connection_id, _ = attached_signal
    async with pool.acquire() as conn:
        await queries.archive_connection(conn, connection_id, account_id="acc_owner")
    submit = await _unregister(pool, PHONE, "acc_owner")
    assert submit.await_count == 1


async def test_other_tenants_attachment_does_not_block(
    attached_signal: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, _, _ = attached_signal
    submit = await _unregister(pool, PHONE, "acc_other")
    assert submit.await_count == 1
