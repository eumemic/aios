"""Integration tests: ``services.confirm_tool_deny`` is idempotent on
``(session_id, tool_call_id)``.

The deny path appends a ``role:"tool"`` event (with ``is_error=True``
and a synthesized rejection message) so the model sees the rejection
in its next context window. Pre-fix it forwarded straight to
``append_event``, so a retried POST
``/v1/sessions/{id}/tool-confirmations`` with ``decision="deny"``
appended a *second* ``role:"tool"`` event with the same
``tool_call_id``.

The bug shape is identical to the ``append_tool_result`` case fixed in
PR #445: ``harness/context.py:499-506`` builds ``real_results: dict[tcid
→ data]`` by iterating and overwriting, so a duplicate with different
content (e.g. an operator who re-clicked deny with a different
``deny_message``) silently rewrites the model's view of an earlier
turn — direct violation of the monotonic-context invariant.

This file pins the same idempotency contract for ``confirm_tool_deny``.
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
    entry. ``confirm_tool_deny`` finds the tool name via
    ``read_message_events`` + ``_find_tool_call``."""
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
            pool, account_id="acc_test", prefix="deny-test", tools=[ToolSpec(type="bash")]
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
                            "id": "tc_deny_test",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
            )
        yield pool, "acc_test", session.id, "tc_deny_test"
    finally:
        await pool.close()


async def _count_tool_role_events(
    pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str
) -> int:
    async with pool.acquire() as conn:
        return (
            await conn.fetchval(
                """
                SELECT COUNT(*) FROM events
                 WHERE session_id = $1
                   AND kind = 'message'
                   AND data->>'role' = 'tool'
                   AND data->>'tool_call_id' = $2
                """,
                session_id,
                tool_call_id,
            )
            or 0
        )


class TestConfirmToolDenyIdempotency:
    async def test_duplicate_deny_does_not_create_second_event(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """Two denies on the same ``tool_call_id`` must produce ONE event.

        Today the second call appends a duplicate ``role:"tool"`` event
        and the monotonic-context invariant is violated (`real_results`
        in ``context.py`` overwrites).
        """
        pool, account_id, session_id, tool_call_id = session_with_parent_tool_call

        await sessions_service.confirm_tool_deny(
            pool,
            session_id,
            tool_call_id,
            "first rejection",
            account_id=account_id,
        )
        await sessions_service.confirm_tool_deny(
            pool,
            session_id,
            tool_call_id,
            "second rejection",
            account_id=account_id,
        )

        count = await _count_tool_role_events(pool, session_id, tool_call_id)
        assert count == 1, (
            f"duplicate deny appended a second tool-role event "
            f"(count={count}); monotonic-context invariant violated"
        )

    async def test_duplicate_deny_returns_original_content(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """The idempotent return must carry the *first* call's rejection
        message. Returning the duplicate's content would silently corrupt
        the prior history."""
        pool, account_id, session_id, tool_call_id = session_with_parent_tool_call

        await sessions_service.confirm_tool_deny(
            pool,
            session_id,
            tool_call_id,
            "first rejection",
            account_id=account_id,
        )
        second_event = await sessions_service.confirm_tool_deny(
            pool,
            session_id,
            tool_call_id,
            "second rejection",
            account_id=account_id,
        )

        assert "first rejection" in second_event.data["content"], (
            f"second-call return carries the wrong content "
            f"(data={second_event.data!r}); idempotent return must "
            f"preserve the first-call's truth"
        )
