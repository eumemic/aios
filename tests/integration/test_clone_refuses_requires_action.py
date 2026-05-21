"""``clone_session`` must refuse a parent parked in ``requires_action``:
the clone inherits ``stop_reason`` verbatim, so the operator action the
parent awaits would never reach the clone's URL."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def requires_action_parent(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, parent_id)`` for an idle session
    parked in ``requires_action`` with a custom-tool pending."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_clone_ra', NULL, TRUE, 'clone-requires-action-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_clone_ra", prefix="clone-ra"
        )
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                session_id=session.id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_custom",
                            "type": "custom",
                            "function": {"name": "ask_user", "arguments": "{}"},
                        }
                    ],
                },
                account_id="acc_clone_ra",
            )
            await queries.set_session_status(
                conn,
                session.id,
                "idle",
                {"type": "requires_action", "custom_tools": ["tc_custom"]},
                account_id="acc_clone_ra",
            )
        yield pool, "acc_clone_ra", session.id
    finally:
        await pool.close()


async def test_clone_refuses_requires_action_parent(
    requires_action_parent: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, parent_id = requires_action_parent

    with pytest.raises(ConflictError) as excinfo:
        await sessions_service.clone_session(pool, parent_id, account_id=account_id)

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("id") == parent_id

    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE id <> $1 AND account_id = $2",
            parent_id,
            account_id,
        )
    assert n == 0, f"clone row leaked despite refusal: {n} extra sessions"
