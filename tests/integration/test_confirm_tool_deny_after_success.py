"""Integration test: ``confirm_tool_deny`` must distinguish a
genuine retry of a prior deny from a deny that arrives AFTER the
tool already produced a success result.

Pre-fix ``append_tool_result``'s idempotency check uses
``find_tool_result_event`` which matches ANY tool-role event for
``tool_call_id``, success OR error. So:

  1. Tool fires (e.g., always_allow agent, or pre-#533 bogus allow,
     or two-tab race where one allow landed before the other deny).
  2. The successful tool result event lands on the event log.
  3. Operator clicks deny. ``confirm_tool_deny`` →
     ``append_tool_result(is_error=True)`` → finds the success event,
     treats the deny as an idempotent retry, returns the SUCCESS event.

The HTTP response is 201 with ``data.content`` = the tool's output.
The model's context still carries the success result. The deny had
no effect: tool DID run, model sees the result, user believed they
denied.

The dedup-by-tool_call_id contract is broken when the existing
event's intent differs from the new request's intent. Correct
behavior: idempotent retry requires matching ``is_error``; an
attempted deny against an already-succeeded tool is a CONFLICT,
not a retry. Surface as ``ConflictError`` (router → 409); operator
learns "too late, tool ran" rather than silent success.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_with_tool_already_succeeded(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, session_id, tool_call_id)`` for a
    session whose event log contains: assistant message with a
    tool_calls entry, followed by a SUCCESSFUL tool-role result
    event for that same tool_call_id."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_deny_race', NULL, TRUE, 'deny-after-success-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id="acc_deny_race",
            prefix="deny-race",
            tools=[ToolSpec(type="bash")],
        )
        async with pool.acquire() as conn:
            # Parent assistant message with the tool_call.
            await queries.append_event(
                conn,
                account_id="acc_deny_race",
                session_id=session.id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_already_ran",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
            )
            # The tool ran successfully (some path bypassed the
            # always_ask gate — two-tab race, always_allow tool,
            # bogus pre-confirm pre-#533, etc.) and produced a
            # success result event.
            await queries.append_event(
                conn,
                account_id="acc_deny_race",
                session_id=session.id,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": "tc_already_ran",
                    "content": "the-tool-actually-executed",
                    "name": "bash",
                    # NOTE: no is_error key → success result.
                },
            )
        yield pool, "acc_deny_race", session.id, "tc_already_ran"
    finally:
        await pool.close()


class TestConfirmToolDenyAfterSuccess:
    async def test_deny_after_success_raises_conflict_not_returns_success(
        self,
        session_with_tool_already_succeeded: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """A deny for a tool_call_id whose tool already produced a
        successful result must NOT be treated as an idempotent retry —
        the existing event's intent doesn't match the new request's
        intent. Surface as ``ConflictError`` so the operator learns
        the deny is too late rather than silently receiving the
        success event back."""
        pool, account_id, session_id, tool_call_id = session_with_tool_already_succeeded

        with pytest.raises(ConflictError):
            await sessions_service.confirm_tool_deny(
                pool,
                session_id,
                tool_call_id,
                "operator intended to deny",
                account_id=account_id,
            )

        # And no second tool-role event for this tool_call_id should have
        # been appended (we still want the single, true success event in
        # the log — the conflict refused to write, not to coexist).
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM events
                 WHERE session_id = $1
                   AND data->>'role' = 'tool'
                   AND data->>'tool_call_id' = $2
                """,
                session_id,
                tool_call_id,
            )
        assert count == 1, (
            f"deny-after-success must not append a duplicate tool-role event; current count={count}"
        )

    async def test_deny_idempotent_when_prior_deny_exists(
        self,
        migrated_db_url: str,
        _reset_db_state: None,
    ) -> None:
        """Regression: a deny retry for an already-denied tool_call still
        returns the original deny event (idempotency contract from #447
        preserved). The conflict guard only fires for intent mismatch,
        not same-intent retries."""
        pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                    VALUES ('acc_deny_retry', NULL, TRUE, 'deny-retry-test')
                    """
                )
            _agent, _env, session = await seed_agent_env_session(
                pool,
                account_id="acc_deny_retry",
                prefix="deny-retry",
                tools=[ToolSpec(type="bash")],
            )
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn,
                    account_id="acc_deny_retry",
                    session_id=session.id,
                    kind="message",
                    data={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "tc_to_deny",
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{}"},
                            }
                        ],
                    },
                )

            first = await sessions_service.confirm_tool_deny(
                pool, session.id, "tc_to_deny", "no", account_id="acc_deny_retry"
            )
            second = await sessions_service.confirm_tool_deny(
                pool, session.id, "tc_to_deny", "no", account_id="acc_deny_retry"
            )
            assert first.id == second.id, "same-intent retry must return original event"
        finally:
            await pool.close()
