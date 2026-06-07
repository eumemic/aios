"""Integration test for #741: ``Session.awaiting`` must surface unresolved
tool_calls from EVERY assistant turn, not just the latest.

``_unresolved_tool_calls`` (backing ``list_unresolved_tool_calls_batch`` →
``compute_awaiting`` → ``Session.awaiting``) previously used
``SELECT DISTINCT ON (session_id) ... ORDER BY seq DESC`` — the single latest
assistant message with non-empty tool_calls.  A tool_call on an EARLIER
assistant whose result hasn't landed was invisible to ``awaiting`` whenever a
LATER assistant also emitted tool_calls.  Sibling of #737 on the read-model
side: after #737 dispatch runs such a tool, but the awaiting view couldn't see
it.

Reachable scenario (the one #737 documents): A1 emits an ``always_ask``
tool_call X; before the operator confirms, an impatient user lifts the session
out of ``requires_action``; the model emits A2 with a different tool_call Z.
A2 is now the latest assistant-with-tool_calls, so ``awaiting`` reported only Z
and omitted the still-pending X.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.agents import ToolSpec
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


def _assistant(tool_call_id: str, name: str = "bash") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


@pytest.fixture
async def session_with_two_assistant_turns(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a session whose log is:
    A1(tool_call tc_X) → user → A2(tool_call tc_Z), with NO tool_result for
    either.  ``tc_X`` is the earlier, non-latest unresolved tool_call.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_awaiting_all_assistants"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "awaiting-all-assistants-test",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="awaiting-all-assistants",
            tools=[ToolSpec(type="bash")],
        )
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data=_assistant("tc_X"),
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data={"role": "user", "content": "are you still there?"},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data=_assistant("tc_Z"),
            )
        yield pool, account_id, session.id
    finally:
        await pool.close()


class TestAwaitingSpansAllAssistants:
    async def test_unresolved_from_earlier_assistant_is_surfaced(
        self,
        session_with_two_assistant_turns: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """#741: an unresolved tool_call on a non-latest assistant must appear
        in the unresolved batch, not be hidden by a later assistant turn."""
        pool, account_id, session_id = session_with_two_assistant_turns
        async with pool.acquire() as conn:
            unresolved = await queries.list_unresolved_tool_calls_batch(
                conn, [session_id], account_id=account_id
            )
        tcids = {e["tool_call_id"] for e in unresolved.get(session_id, [])}
        assert tcids == {"tc_X", "tc_Z"}, (
            "Session.awaiting missed tc_X (on the earlier assistant A1) because "
            "the unresolved-tool_calls query only inspected the latest assistant "
            "turn (#741, sibling of #737)"
        )
