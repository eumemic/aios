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
from aios.harness.sweep import GHOST_ASST_SQL, find_and_repair_ghosts
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


async def test_ghost_asst_sql_excludes_errored_session(
    errored_session_with_tool_call: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """``GHOST_ASST_SQL`` itself must drop the errored session's
    assistant-with-tool_calls rows — the errored exclusion is pushed into
    SQL (#897), not only applied by the Python ``_errored_session_ids``
    post-filter. The errored session here has an open tool_call ``tc_x``
    (``open_tool_call_count > 0``), so without the pushed-down
    ``last_error_seq`` predicate the row would survive the scan and be
    fetched from the event log on every 30s sweep only to be discarded."""
    pool, _account_id, session_id = errored_session_with_tool_call

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            GHOST_ASST_SQL.format(scope_clause="AND e.session_id = $1"),
            session_id,
        )
    assert rows == [], (
        f"GHOST_ASST_SQL returned rows for an errored session (got {len(rows)}); "
        f"the errored-session exclusion (#897) was not pushed into SQL."
    )


@pytest.fixture
async def healthy_session_with_open_tool_call(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a NON-errored session
    carrying an assistant ``tool_call`` (``tc_live``) for the registered
    built-in ``bash`` tool with no matching result — a genuine ghost.

    This is the positive counterpart to the errored fixture: it pins that
    pushing the errored predicate into ``GHOST_ASST_SQL`` does NOT
    over-exclude a healthy errored-adjacent session. The session has
    ``open_tool_call_count > 0`` but ``last_error_seq`` is unset, so it must
    survive the scan and the ghost must be repaired."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_live_tool', NULL, TRUE, 'healthy-open-call')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_live_tool", prefix="live-tool"
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
                            "id": "tc_live",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
                account_id="acc_live_tool",
            )
        yield pool, "acc_live_tool", session.id
    finally:
        await pool.close()


async def test_ghost_repair_handles_healthy_session_with_open_call(
    healthy_session_with_open_tool_call: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A non-errored session with an open dispatched tool_call still has its
    ghost repaired (#897). Pushing the errored predicate into
    ``GHOST_ASST_SQL`` must not drop a session that ``ERRORED_SESSIONS_SQL``
    would NOT classify as errored — otherwise a genuine ghost would never be
    repaired."""
    pool, _account_id, session_id = healthy_session_with_open_tool_call

    # The session's assistant-with-tool_calls row must survive the scan.
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            GHOST_ASST_SQL.format(scope_clause="AND e.session_id = $1"),
            session_id,
        )
    assert len(rows) == 1, (
        f"GHOST_ASST_SQL dropped a healthy (non-errored) session with an open "
        f"tool_call (got {len(rows)} rows); the #897 predicate over-excludes."
    )

    repaired = await find_and_repair_ghosts(pool, TaskRegistry(), session_id=session_id)
    assert (session_id, "tc_live") in repaired, (
        f"ghost-repair missed the open dispatched tool_call on a healthy "
        f"session (got {repaired}); the #897 errored exclusion must not "
        f"over-exclude non-errored sessions."
    )
