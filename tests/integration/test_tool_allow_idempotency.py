"""Integration tests: ``services.confirm_tool_allow`` is idempotent on
``(session_id, tool_call_id)``.

The allow path appends a ``lifecycle/tool_confirmed`` event that the
worker's step function picks up to dispatch the tool call. Pre-fix it
forwarded straight to ``append_event``, so a retried POST
``/v1/sessions/{id}/tool-confirmations`` with ``decision="allow"`` (a
double-click, an SSE-disconnect retry, a network blip) appended a
*second* ``lifecycle/tool_confirmed`` event with the same
``tool_call_id`` — the unfixed sibling of the deny twin's #447 fix.

Mitigated downstream by ``_dispatch_confirmed_tools`` collecting into a
set (so the tool dispatches once), but the lifecycle event log
accumulates duplicate rows — and the deny twin's own docstring spells
out the contract this asymmetry violates ("Idempotent on
``(session_id, tool_call_id)``: a retried POST returns the original
event").

This file pins the same idempotency contract for ``confirm_tool_allow``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_with_parent_tool_call(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, session_id, tool_call_id)`` for an
    initialized session with one assistant event carrying a ``tool_calls``
    entry."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_test', NULL, TRUE, 'tenant-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_test", prefix="allow-test", tools=[ToolSpec(type="bash")]
        )
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id="acc_test",
                session_id=session.id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_allow_test",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
            )
        yield pool, "acc_test", session.id, "tc_allow_test"
    finally:
        await pool.close()


async def _count_tool_confirmed_events(
    pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str
) -> int:
    async with pool.acquire() as conn:
        return (
            await conn.fetchval(
                """
                SELECT COUNT(*) FROM events
                 WHERE session_id = $1
                   AND kind = 'lifecycle'
                   AND data->>'event' = 'tool_confirmed'
                   AND data->>'tool_call_id' = $2
                """,
                session_id,
                tool_call_id,
            )
            or 0
        )


class TestConfirmToolAllowIdempotency:
    async def test_duplicate_allow_does_not_create_second_event(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """Two allows on the same ``tool_call_id`` must produce ONE event."""
        pool, account_id, session_id, tool_call_id = session_with_parent_tool_call

        await sessions_service.confirm_tool_allow(
            pool, session_id, tool_call_id, account_id=account_id
        )
        await sessions_service.confirm_tool_allow(
            pool, session_id, tool_call_id, account_id=account_id
        )

        count = await _count_tool_confirmed_events(pool, session_id, tool_call_id)
        assert count == 1, (
            f"duplicate allow appended a second lifecycle/tool_confirmed event "
            f"(count={count}); asymmetric with the deny twin's idempotency contract"
        )

    async def test_duplicate_allow_returns_original_event(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """The idempotent return must be the *first* call's event — same
        id, same seq — so a retried POST gives the caller a stable handle."""
        pool, account_id, session_id, tool_call_id = session_with_parent_tool_call

        first = await sessions_service.confirm_tool_allow(
            pool, session_id, tool_call_id, account_id=account_id
        )
        second = await sessions_service.confirm_tool_allow(
            pool, session_id, tool_call_id, account_id=account_id
        )

        assert second.id == first.id
        assert second.seq == first.seq
