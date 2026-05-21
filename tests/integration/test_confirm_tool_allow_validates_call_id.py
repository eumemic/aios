"""Integration test: ``services.confirm_tool_allow`` must reject
bogus ``tool_call_id`` values — values that don't correspond to any
real assistant tool_call in the session's event log.

Pre-fix the allow path locked the session row and appended a
``lifecycle/tool_confirmed`` event without verifying that the
``tool_call_id`` named in the request matches a tool_call the
model actually emitted. The deny twin's path (via
``append_tool_result`` → ``lookup_tool_name_by_call_id``)
explicitly raises ``NotFoundError`` on a missing tool_call — so
this is an asymmetric validation gap that violates CLAUDE.md's
"fail hard, no fallbacks" stance.

Downstream consequences:
- Log poisoning: any authenticated API client can fill a session's
  lifecycle event stream with arbitrary ``tool_confirmed`` rows.
- ``_dispatch_confirmed_tools`` (``harness/loop.py``) maintains
  ``confirmed: set[str]`` from the latest assistant message and
  matches it against tc.get("id") in confirmed. A pre-emptive
  allow for an unguessable id today is hopeless, but provider
  tool_call_id generation is provider-controlled and may converge
  on guessable patterns (sequential, hashed, prefix-deterministic),
  potentially enabling pre-confirmation bypass of the always_ask
  gate.

The fix mirrors deny: call ``lookup_tool_name_by_call_id`` (or
equivalent) to verify the tool_call exists in the session, before
appending the confirmation event.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_with_parent_tool_call(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, session_id, real_tool_call_id)`` for a
    session whose event log contains exactly one assistant tool_calls
    entry with id ``tc_real``."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_validate', NULL, TRUE, 'allow-validate-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_validate", prefix="allow-validate", tools=[ToolSpec(type="bash")]
        )
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id="acc_validate",
                session_id=session.id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_real",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
            )
        yield pool, "acc_validate", session.id, "tc_real"
    finally:
        await pool.close()


class TestConfirmToolAllowValidatesCallId:
    async def test_allow_rejects_bogus_tool_call_id(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """A confirm-allow POST for a tool_call_id that doesn't appear
        in any assistant message in the session must raise
        ``NotFoundError`` (the same surface as the deny twin's missing-
        call case) rather than silently appending a
        ``lifecycle/tool_confirmed`` row that pollutes the event log."""
        pool, account_id, session_id, real_tool_call_id = session_with_parent_tool_call
        bogus = "tc_does_not_exist_anywhere"
        assert bogus != real_tool_call_id

        with pytest.raises(NotFoundError):
            await sessions_service.confirm_tool_allow(
                pool, session_id, bogus, account_id=account_id
            )

        # And the bogus tool_call_id MUST NOT have appended a lifecycle
        # event — pre-fix the lock + append fires before any validation.
        async with pool.acquire() as conn:
            poisoned = await conn.fetchval(
                """
                SELECT COUNT(*) FROM events
                 WHERE session_id = $1
                   AND kind = 'lifecycle'
                   AND data->>'event' = 'tool_confirmed'
                   AND data->>'tool_call_id' = $2
                """,
                session_id,
                bogus,
            )
        assert poisoned == 0, (
            f"confirm_tool_allow with bogus tool_call_id wrote "
            f"{poisoned} lifecycle event(s) to the session log. The "
            f"validation must happen BEFORE the append; rejecting at "
            f"the response layer but still writing the event leaves "
            f"durable poison in the log."
        )

    async def test_allow_succeeds_for_real_tool_call_id(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """Happy-path regression: a real tool_call_id still produces a
        valid ``tool_confirmed`` event."""
        pool, account_id, session_id, real_tool_call_id = session_with_parent_tool_call

        event = await sessions_service.confirm_tool_allow(
            pool, session_id, real_tool_call_id, account_id=account_id
        )
        assert event.data["event"] == "tool_confirmed"
        assert event.data["tool_call_id"] == real_tool_call_id
        assert event.data["result"] == "allow"
