"""Tool-result and tool-confirmation POSTs on an ``errored`` session
must raise ConflictError instead of silently appending events the
sweep ignores.  Ghost-repair skips errored sessions for the same
reason ``CANDIDATE_ROWS_SQL`` / ``CONFIRMED_ROWS_SQL`` already do."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.harness.sweep import find_and_repair_ghosts
from aios.harness.task_registry import TaskRegistry
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def errored_session_with_tool_call(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for an errored session
    that has an assistant message carrying a tool_call ``tc_x``.  The
    tool_call has no matching tool-role result, so it would be a
    ghost candidate for the sweep."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_err_tool', NULL, TRUE, 'errored-tool-actions')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_err_tool", prefix="err-tool"
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
                            "id": "tc_x",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
                account_id="acc_err_tool",
            )
            # Park the session in the derived ``errored`` state: a
            # ``turn_ended``/``error`` lifecycle event with no later user
            # message (see queries._SESSION_ERRORED_EXPR / sweep).
            await queries.append_event(
                conn,
                session_id=session.id,
                kind="lifecycle",
                data={"event": "turn_ended", "status": "errored", "stop_reason": "error"},
                account_id="acc_err_tool",
            )
            await queries.set_session_stop_reason(
                conn, session.id, {"type": "error"}, account_id="acc_err_tool"
            )
        yield pool, "acc_err_tool", session.id
    finally:
        await pool.close()


async def test_append_tool_result_rejects_errored_session(
    errored_session_with_tool_call: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, session_id = errored_session_with_tool_call

    async with pool.acquire() as conn:
        with pytest.raises(ConflictError) as excinfo:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id="tc_x",
                content="result",
                is_error=False,
            )
    assert excinfo.value.detail is not None
    assert excinfo.value.detail.get("session_id") == session_id

    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND role = 'tool'",
            session_id,
        )
    assert n == 0, f"a tool-role event leaked onto the errored row (count={n})"


async def test_confirm_tool_allow_rejects_errored_session(
    errored_session_with_tool_call: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, session_id = errored_session_with_tool_call

    with pytest.raises(ConflictError) as excinfo:
        await sessions_service.confirm_tool_allow(pool, session_id, "tc_x", account_id=account_id)
    assert excinfo.value.detail is not None
    assert excinfo.value.detail.get("session_id") == session_id

    async with pool.acquire() as conn:
        n = await conn.fetchval(
            """
            SELECT count(*) FROM events
             WHERE session_id = $1
               AND kind = 'lifecycle'
               AND data->>'event' = 'tool_confirmed'
            """,
            session_id,
        )
    assert n == 0, f"a tool_confirmed lifecycle event leaked onto the errored row (count={n})"


async def test_confirm_tool_deny_rejects_errored_session(
    errored_session_with_tool_call: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """``confirm_tool_deny`` delegates to ``append_tool_result``; pin the
    contract directly so a future refactor that bypasses the delegation
    is caught here, not at runtime."""
    pool, account_id, session_id = errored_session_with_tool_call

    with pytest.raises(ConflictError):
        await sessions_service.confirm_tool_deny(
            pool, session_id, "tc_x", "user denied", account_id=account_id
        )

    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND role = 'tool'",
            session_id,
        )
    assert n == 0


async def test_ghost_repair_skips_errored_session(
    errored_session_with_tool_call: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Ghost-repair must skip derived-errored sessions; otherwise it would
    trip the operator-facing ConflictError on every errored session and abort
    the sweep loop."""
    pool, _account_id, session_id = errored_session_with_tool_call

    repaired = await find_and_repair_ghosts(pool, TaskRegistry(), session_id=session_id)
    assert repaired == [], (
        f"ghost-repair attempted on an errored session (got {repaired}); "
        f"find_and_repair_ghosts is missing the derived-errored exclusion "
        f"(_errored_session_ids)."
    )
