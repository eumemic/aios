"""Integration test: ``confirm_tool_allow`` must reject when the
referenced tool_call has already been resolved (a tool-role result
already exists in the log).

Pre-fix the allow path checks only for an existing
``lifecycle/tool_confirmed`` event (idempotent retry guard).  It does
NOT check whether a tool-role result already exists for the same
``tool_call_id``.  So:

  1. Tool fires (always_ask).  Session parks in ``requires_action``.
  2. Operator denies in tab A → ``confirm_tool_deny`` →
     ``append_tool_result(is_error=True)``.  A tool-role error event
     lands on the log; no ``lifecycle/tool_confirmed`` event exists.
  3. Operator clicks allow in tab B → ``confirm_tool_allow`` →
     ``find_tool_confirmed_event`` returns ``None`` → the allow
     appends a fresh ``lifecycle/tool_confirmed allow`` event.

The HTTP response is 201 with the lifecycle event.  Operator UI
displays "allow accepted."  But the model's context still carries the
deny's error result, has reacted to it, and may have moved on.  The
allow had no effect.  A textbook silent-failure: confirming bool
returns success while the underlying intent is impossible.

Symmetric twin to #535 (``confirm_tool_deny`` rejected when tool
already succeeded).  Same defect class: "confirm endpoints silently
accept impossible inputs."  The contract should be: idempotent retry
requires no prior result; an attempted allow against an
already-resolved tool is a CONFLICT, not a no-op.

The same trap applies to allow-after-success (always_allow tool, or
any path that produced a result before the allow arrived).  Both
paths are covered.
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
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service

pytestmark = pytest.mark.integration


async def _seed_session_with_tool_call(
    pool: asyncpg.Pool[Any], account_id: str, suffix: str
) -> tuple[str, str]:
    """Create an account, agent, env, session, and append one assistant
    event carrying a single ``tool_calls`` entry.  Returns
    ``(session_id, tool_call_id)``.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ($1, NULL, TRUE, $2)
            """,
            account_id,
            f"allow-after-resolved-test-{suffix}",
        )
    agent = await agents_service.create_agent(
        pool,
        account_id=account_id,
        name=f"allow-after-resolved-agent-{suffix}",
        model="openrouter/test",
        system="",
        tools=[ToolSpec(type="bash")],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    env = await environments_service.create_environment(
        pool, account_id=account_id, name=f"allow-after-resolved-env-{suffix}"
    )
    tool_call_id = f"tc_resolved_{suffix}"
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            account_id=account_id,
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title=None,
            metadata={},
        )
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session.id,
            kind="message",
            data={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
        )
    return session.id, tool_call_id


@pytest.fixture
async def session_with_tool_already_denied(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, session_id, tool_call_id)`` for a
    session whose event log contains: assistant message with a
    ``tool_calls`` entry, followed by a tool-role ERROR result event
    for that same ``tool_call_id`` (i.e. the deny twin already ran).
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_allow_after_deny"
        session_id, tool_call_id = await _seed_session_with_tool_call(pool, account_id, "deny")
        # Simulate a prior deny: appends a tool-role error event for
        # the same ``tool_call_id`` (mirrors what
        # ``confirm_tool_deny`` would write).
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": '{"error": "Permission denied by operator."}',
                    "name": "bash",
                    "is_error": True,
                },
            )
        yield pool, account_id, session_id, tool_call_id
    finally:
        await pool.close()


@pytest.fixture
async def session_with_tool_already_succeeded(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, session_id, tool_call_id)`` for a
    session whose event log contains: assistant message with a
    ``tool_calls`` entry, followed by a SUCCESSFUL tool-role result
    event for that same ``tool_call_id`` (always_allow tool or
    pre-confirm bypass race produced a result before this allow).
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_allow_after_success"
        session_id, tool_call_id = await _seed_session_with_tool_call(pool, account_id, "success")
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "the-tool-actually-executed",
                    "name": "bash",
                    # NOTE: no is_error key → success result.
                },
            )
        yield pool, account_id, session_id, tool_call_id
    finally:
        await pool.close()


class TestConfirmToolAllowAfterResolved:
    async def test_allow_after_deny_raises_conflict_not_silent_no_op(
        self,
        session_with_tool_already_denied: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """An allow for a tool_call_id that already has a deny error
        result must NOT silently succeed — the lifecycle event would
        have no dispatch effect (the result already pinned the model's
        view) yet the operator sees 201, believing the allow took.

        Surface as ``ConflictError`` so the operator learns "too late,
        already denied" rather than receiving a phantom success.
        """
        pool, account_id, session_id, tool_call_id = session_with_tool_already_denied

        with pytest.raises(ConflictError):
            await sessions_service.confirm_tool_allow(
                pool, session_id, tool_call_id, account_id=account_id
            )

        # No ``lifecycle/tool_confirmed`` event should have been written —
        # the conflict refused the append, didn't merely coexist with it.
        async with pool.acquire() as conn:
            count = await conn.fetchval(
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
        assert count == 0, (
            f"allow-after-deny must not append a phantom lifecycle event; current count={count}"
        )

    async def test_allow_after_success_raises_conflict_not_silent_no_op(
        self,
        session_with_tool_already_succeeded: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """An allow for a tool_call_id that already produced a success
        result must NOT silently succeed.  The tool already ran (e.g.
        always_allow toolset, or any race where dispatch beat the
        operator click); confirming "allow" can't go back in time.

        Same defect shape as deny-after-success (#535) — confirm
        endpoint must reject impossible inputs rather than return a
        misleading 201.
        """
        pool, account_id, session_id, tool_call_id = session_with_tool_already_succeeded

        with pytest.raises(ConflictError):
            await sessions_service.confirm_tool_allow(
                pool, session_id, tool_call_id, account_id=account_id
            )

        async with pool.acquire() as conn:
            count = await conn.fetchval(
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
        assert count == 0, (
            f"allow-after-success must not append a phantom lifecycle event; current count={count}"
        )
