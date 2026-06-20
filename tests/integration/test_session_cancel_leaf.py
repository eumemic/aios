"""Integration tests for the session-side cancel leaf + the C2 sweep clause (6e).

A cancel-marked session must (a) be selected by ``find_sessions_needing_inference`` even when
idle (so the sweep wakes it), and (b) when its step runs the leaf, answer each marked request
``cancelled`` and harvest the marker (so it does not hot-loop). Owned-session teardown +
recursive propagation are the deferred §4.1/§4.4 residuals — out of scope for this slice.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness.sweep import find_sessions_needing_inference
from aios.harness.task_registry import TaskRegistry
from aios.services import sessions as service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_cancel_leaf"


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'cancel-leaf')",
                _ACCOUNT,
            )
        _agent, env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="cancel-leaf"
        )
        yield pool, session.id, env.id
    finally:
        await pool.close()


async def _open_request(
    pool: asyncpg.Pool[Any], session_id: str, env_id: str, *, request_id: str
) -> None:
    """Open an awaited api-caller request edge on the session (no caller wake needed)."""
    async with pool.acquire() as conn:
        await queries.append_request_opened(
            conn,
            session_id=session_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            caller={"kind": "api", "id": _ACCOUNT},
            depth=0,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            awaited=True,
        )


async def test_cancel_marked_session_is_swept_then_leaf_answers_cancelled(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, session_id, env_id = pool_and_session
    await _open_request(pool, session_id, env_id, request_id="req_c")
    async with pool.acquire() as conn:
        await queries.insert_session_cancel_marker(
            conn, session_id=session_id, request_id="req_c", account_id=_ACCOUNT
        )

    # C2: the marked session is selected even though it is otherwise idle (no unreacted msgs).
    needs = await find_sessions_needing_inference(pool, TaskRegistry(), session_id=session_id)
    assert session_id in needs

    # The leaf answers the request cancelled + harvests the marker.
    assert await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)

    async with pool.acquire() as conn:
        resolved = await queries.derive_response(
            conn, session_id, account_id=_ACCOUNT, request_id="req_c"
        )
        assert resolved == {"result": None, "is_error": True, "error": {"kind": "cancelled"}}
        # The request is closed (answered), and the marker is harvested → no re-wake.
        assert await queries.get_open_request_ids(conn, session_id, account_id=_ACCOUNT) == []
        marker = await queries.get_session_cancel_marker(
            conn, session_id=session_id, request_id="req_c"
        )
        assert marker is not None and marker.harvested_at is not None

    # Idempotent: with the marker harvested, a second leaf run is a no-op (no hot-loop).
    assert not await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)
    needs_again = await find_sessions_needing_inference(pool, TaskRegistry(), session_id=session_id)
    assert session_id not in needs_again


async def test_unmarked_session_runs_no_leaf(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A session with an open request but NO cancel-marker is untouched by the leaf."""
    pool, session_id, env_id = pool_and_session
    await _open_request(pool, session_id, env_id, request_id="req_live")
    assert not await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)
    async with pool.acquire() as conn:
        assert await queries.get_open_request_ids(conn, session_id, account_id=_ACCOUNT) == [
            "req_live"
        ]
