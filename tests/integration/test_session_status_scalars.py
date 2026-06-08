"""Integration tests for the four monotonic session-status scalar columns.

The columns — ``last_reacted_seq``, ``open_tool_call_count``,
``last_error_seq``, ``last_user_seq`` — are maintained transactionally
inside ``append_event`` and replace the O(n) correlated-subquery status
derivation with pure column arithmetic.

Tests exercise both the scalar maintenance (column values after specific
event sequences) and the derived ``status`` predicate (active/idle from
column arithmetic). Written TDD-first: they must fail before the
migration and ``append_event`` changes land.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a fresh session."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_scalars', NULL, TRUE, 'scalars-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_scalars", prefix="scalars-test"
        )
        yield pool, "acc_scalars", session.id
    finally:
        await pool.close()


async def _scalars(conn: asyncpg.Connection[Any], session_id: str) -> dict[str, int]:
    """Read the four scalar columns directly from the sessions row."""
    row = await conn.fetchrow(
        "SELECT last_reacted_seq, open_tool_call_count, "
        "last_error_seq, last_user_seq "
        "FROM sessions WHERE id = $1",
        session_id,
    )
    assert row is not None
    return dict(row)


# ─── scalar maintenance tests ────────────────────────────────────────────────


class TestScalarMaintenance:
    async def test_fresh_session_scalars_zero(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, _account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            s = await _scalars(conn, session_id)
        assert s["last_reacted_seq"] == 0
        assert s["open_tool_call_count"] == 0
        assert s["last_error_seq"] == 0
        assert s["last_user_seq"] == 0

    async def test_user_message_bumps_last_user_seq(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            s = await _scalars(conn, session_id)
        assert s["last_user_seq"] == 1

    async def test_assistant_with_reacting_to(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "hi", "reacting_to": 1},
            )
            s = await _scalars(conn, session_id)
        assert s["last_reacted_seq"] == 1

    async def test_turn_ended_bumps_last_reacted_seq(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "hi", "reacting_to": 1},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={"event": "turn_ended"},
            )
            s = await _scalars(conn, session_id)
        # turn_ended is seq 3; last_reacted_seq should be 3
        assert s["last_reacted_seq"] == 3

    async def test_assistant_with_tool_calls_increments_open_count(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                        {
                            "id": "tc_2",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                    ],
                },
            )
            s = await _scalars(conn, session_id)
        assert s["open_tool_call_count"] == 2

    async def test_tool_result_decrements_open_count(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                        {
                            "id": "tc_2",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                    ],
                },
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            )
            s = await _scalars(conn, session_id)
            assert s["open_tool_call_count"] == 1

            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "tool", "tool_call_id": "tc_2", "content": "ok"},
            )
            s = await _scalars(conn, session_id)
            assert s["open_tool_call_count"] == 0

    async def test_error_lifecycle_sets_last_error_seq(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={"stop_reason": "error"},
            )
            s = await _scalars(conn, session_id)
        assert s["last_error_seq"] == 1

    async def test_user_message_clears_error(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={"stop_reason": "error"},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "retry"},
            )
            s = await _scalars(conn, session_id)
        assert s["last_user_seq"] > s["last_error_seq"]


# ─── derived status tests ────────────────────────────────────────────────────


class TestDerivedStatus:
    async def test_status_idle_fresh(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            session = await queries.get_session(conn, session_id, account_id=account_id)
        assert session.status == "idle"

    async def test_status_active_after_user(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            session = await queries.get_session(conn, session_id, account_id=account_id)
        assert session.status == "active"

    async def test_status_idle_after_full_turn(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "hi", "reacting_to": 1},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={"event": "turn_ended"},
            )
            session = await queries.get_session(conn, session_id, account_id=account_id)
        assert session.status == "idle"

    async def test_status_active_with_open_tools(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                    ],
                },
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={"event": "turn_ended"},
            )
            session = await queries.get_session(conn, session_id, account_id=account_id)
        assert session.status == "active"

    async def test_status_idle_after_tools_resolved_and_reacted(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            # assistant with tool call
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                    ],
                },
            )
            # tool result
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            )
            # assistant reacts
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "done", "reacting_to": 2},
            )
            # turn ended
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={"event": "turn_ended"},
            )
            session = await queries.get_session(conn, session_id, account_id=account_id)
        assert session.status == "idle"

    async def test_status_active_tools_resolved_before_reaction(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """Tool results land after turn_ended but before the model reacts
        — last_event_seq > last_reacted_seq so the session is active."""
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            # assistant with tool call
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                    ],
                },
            )
            # turn ended
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={"event": "turn_ended"},
            )
            # tool result arrives after turn_ended
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            )
            session = await queries.get_session(conn, session_id, account_id=account_id)
        # last_event_seq=3 (tool result) > last_reacted_seq=2 (turn_ended)
        # AND open_tool_call_count went 1 -> 0 (but last_event_seq > last_reacted_seq)
        assert session.status == "active"

    async def test_errored_hides_active(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """An errored session reads as idle, not active."""
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            # Error lifecycle — normally this would mean active (unreacted user),
            # but errored overrides.
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={"stop_reason": "error"},
            )
            session = await queries.get_session(conn, session_id, account_id=account_id)
        assert session.status == "idle"


# ─── clone + list filter tests ───────────────────────────────────────────────


class TestCloneAndListFilter:
    async def test_clone_copies_scalars(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            # Create a complete turn so the session is idle and cloneable,
            # with non-zero scalars.
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "hi", "reacting_to": 1},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={"event": "turn_ended"},
            )

            parent_scalars = await _scalars(conn, session_id)

            clone = await queries.clone_session(conn, session_id, account_id=account_id)
            clone_scalars = await _scalars(conn, clone.id)

        assert clone_scalars["last_reacted_seq"] == parent_scalars["last_reacted_seq"]
        assert clone_scalars["open_tool_call_count"] == parent_scalars["open_tool_call_count"]
        assert clone_scalars["last_error_seq"] == parent_scalars["last_error_seq"]
        assert clone_scalars["last_user_seq"] == parent_scalars["last_user_seq"]

    async def test_list_sessions_status_filter(
        self,
        migrated_db_url: str,
        _reset_db_state: None,
    ) -> None:
        pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                    VALUES ('acc_filter', NULL, TRUE, 'filter-test')
                    """
                )
            _agent, _env, sess_idle = await seed_agent_env_session(
                pool, account_id="acc_filter", prefix="filter-idle"
            )
            _agent2, _env2, sess_active = await seed_agent_env_session(
                pool, account_id="acc_filter", prefix="filter-active"
            )
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn,
                    account_id="acc_filter",
                    session_id=sess_active.id,
                    kind="message",
                    data={"role": "user", "content": "hello"},
                )

                active_list = await queries.list_sessions(
                    conn, account_id="acc_filter", status="active"
                )
                idle_list = await queries.list_sessions(
                    conn, account_id="acc_filter", status="idle"
                )
            assert {s.id for s in active_list} == {sess_active.id}
            assert sess_idle.id in {s.id for s in idle_list}
            assert sess_active.id not in {s.id for s in idle_list}
        finally:
            await pool.close()


# ─── structural assertion ────────────────────────────────────────────────────


class TestNoEventsSubquery:
    def test_no_events_subquery_in_status(self) -> None:
        """The status expression must use column arithmetic, not event-log
        correlated subqueries."""
        from aios.db.queries import _SESSION_STATUS_EXPR

        assert "FROM events" not in _SESSION_STATUS_EXPR, (
            "_SESSION_STATUS_EXPR still contains a correlated subquery over events"
        )
