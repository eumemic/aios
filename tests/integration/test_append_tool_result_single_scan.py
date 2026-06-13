"""Integration tests (issue #991): ``services.append_tool_result`` performs
AT MOST ONE parent-assistant ``@>`` scan per append, and does not hold the
session ``FOR UPDATE`` across ``append_event``'s tokenizer pass.

Pre-fix the operator/connector/ghost-repair path ran TWO ``@>`` scans against
the same parent-assistant row: ``lookup_tool_name_by_call_id`` (for the
``name`` stamp) and then ``append_event``'s fallback
``_lookup_tool_parent_channel`` (for the derived ``channel``).  The fix has
``lookup_tool_name_by_call_id`` co-select ``focal_channel_at_arrival`` so the
caller can pass ``tool_parent_channel=`` explicitly, collapsing the two scans
to one.  We spy on ``_lookup_tool_parent_channel`` to prove the fallback is
never awaited, mirroring ``test_tool_channel_stamp``'s spy pattern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import events as events_mod
from aios.db.queries import sessions as session_queries
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_with_parent(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, session_id, tool_call_id)`` with one parent
    assistant event carrying a ``tool_calls`` entry on a known focal channel."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_1scan', NULL, TRUE, '1scan-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_1scan", prefix="1scan-test", tools=[ToolSpec(type="bash")]
        )
        async with pool.acquire() as conn:
            await session_queries.set_session_focal_channel(
                conn, session.id, "tg:7", account_id="acc_1scan"
            )
            await queries.append_event(
                conn,
                account_id="acc_1scan",
                session_id=session.id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_scan",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
            )
        yield pool, "acc_1scan", session.id, "tc_scan"
    finally:
        await pool.close()


async def _stored_channel(
    pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str
) -> str | None:
    async with pool.acquire() as conn:
        channel: str | None = await conn.fetchval(
            "SELECT channel FROM events "
            "WHERE session_id = $1 AND data->>'tool_call_id' = $2 "
            "  AND data->>'role' = 'tool' LIMIT 1",
            session_id,
            tool_call_id,
        )
        return channel


class TestSingleParentScan:
    async def test_fallback_parent_lookup_not_awaited(
        self,
        session_with_parent: tuple[asyncpg.Pool[Any], str, str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``append_tool_result`` must NOT trigger the fallback
        ``_lookup_tool_parent_channel`` scan — the parent channel comes from
        the single ``lookup_tool_name_by_call_id`` scan instead."""
        pool, account_id, session_id, tool_call_id = session_with_parent

        spy = AsyncMock(return_value=None)
        monkeypatch.setattr(events_mod, "_lookup_tool_parent_channel", spy)

        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content="ok",
            )

        spy.assert_not_awaited()

    async def test_channel_stamped_from_single_lookup(
        self,
        session_with_parent: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """The tool-role event's derived ``channel`` still equals the parent's
        ``focal_channel_at_arrival`` — the co-selected value is passed through
        as ``tool_parent_channel`` (no behavior change, just one fewer scan)."""
        pool, account_id, session_id, tool_call_id = session_with_parent

        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content="ok",
            )

        assert await _stored_channel(pool, session_id, tool_call_id) == "tg:7"

    async def test_name_stamped_from_parent(
        self,
        session_with_parent: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """The tool's ``name`` is still stamped from the parent's tool_calls."""
        pool, account_id, session_id, tool_call_id = session_with_parent

        async with pool.acquire() as conn:
            event = await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content="ok",
            )
            # The derived ``tool_name`` column is populated from data["name"].
            tool_name = await conn.fetchval("SELECT tool_name FROM events WHERE id = $1", event.id)

        assert event.data["name"] == "bash"
        assert tool_name == "bash"
