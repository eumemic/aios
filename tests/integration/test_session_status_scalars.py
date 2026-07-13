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
        "last_error_seq, last_user_seq, last_stimulus_seq "
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

    async def test_turn_ended_does_not_bump_last_reacted_seq(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """``last_reacted_seq`` tracks the assistant's ``reacting_to`` watermark
        ONLY — ``turn_ended`` lifecycle events must NOT advance it.

        Bumping it on turn_ended (as #749's first cut did) breaks the retry
        loop: a rescheduling ``turn_ended`` appends with no assistant reaction,
        so bumping the watermark falsely marks the unreacted user message as
        reacted-to, flipping a retry-pending session to idle (see
        ``test_litellm_502_recovery``)."""
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
        # The assistant reacted to seq 1; turn_ended (seq 3) does NOT bump it.
        assert s["last_reacted_seq"] == 1

    async def test_assistant_without_reacting_to_uses_own_seq(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """An assistant message with no explicit ``reacting_to`` advances the
        watermark to its OWN seq (mirrors the pre-#732
        ``MAX(COALESCE(reacting_to, seq))`` fallback)."""
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            # assistant reply, no reacting_to → watermark = its own seq (2)
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "hi"},
            )
            s = await _scalars(conn, session_id)
        assert s["last_reacted_seq"] == 2

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

    async def test_status_idle_after_assistant_reply(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """The canonical idle state (#749 regression): a user message followed
        by a plain assistant reply (no tool calls), with NO ``turn_ended``
        lifecycle event yet.

        ``last_event_seq`` (=2, the assistant reply) exceeds
        ``last_reacted_seq`` (=1, what the assistant reacted to), so the OLD
        predicate ``last_event_seq > last_reacted_seq`` reads this as ACTIVE
        and the harness does one extra model step — the bug that fails the
        whole e2e suite. The fix keys the active predicate on
        ``last_stimulus_seq`` (non-assistant messages only): the assistant's
        own reply is not a stimulus, so ``last_stimulus_seq`` (=1) ==
        ``last_reacted_seq`` (=1) → idle.
        """
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
            session = await queries.get_session(conn, session_id, account_id=account_id)
        assert session.status == "idle"

    async def test_status_active_tool_result_after_assistant_reply(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """Over-correction guard for #749: a tool result that lands AFTER a
        plain assistant reply is an unreacted stimulus — the assistant must
        react to it, so the session is ACTIVE.

        This proves ``last_stimulus_seq`` tracks tool messages (role <>
        'assistant'), not just user messages: keying the predicate on
        ``last_user_seq`` instead would wrongly read this idle.
        """
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            # user message (seq 1)
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "run a tool"},
            )
            # assistant reply that reacted to the user (seq 2)
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "hi", "reacting_to": 1},
            )
            # a tool result lands after the reply (seq 3) — an unreacted
            # stimulus the assistant must now react to.
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "tool", "tool_call_id": "tc_late", "content": "ok"},
            )
            session = await queries.get_session(conn, session_id, account_id=account_id)
        # last_stimulus_seq=3 (tool) > last_reacted_seq=1 → active.
        assert session.status == "active"

    async def test_last_stimulus_seq_tracks_user_and_tool_not_assistant(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """``last_stimulus_seq`` advances on user + tool messages but never on
        the assistant's own reply."""
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            # user (seq 1) → last_stimulus_seq = 1
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            row = await conn.fetchrow(
                "SELECT last_stimulus_seq FROM sessions WHERE id = $1", session_id
            )
            assert row is not None and row["last_stimulus_seq"] == 1

            # assistant reply (seq 2) → last_stimulus_seq stays 1
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "hi", "reacting_to": 1},
            )
            row = await conn.fetchrow(
                "SELECT last_stimulus_seq FROM sessions WHERE id = $1", session_id
            )
            assert row is not None and row["last_stimulus_seq"] == 1

            # tool result (seq 3) → last_stimulus_seq advances to 3
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            )
            row = await conn.fetchrow(
                "SELECT last_stimulus_seq FROM sessions WHERE id = $1", session_id
            )
            assert row is not None and row["last_stimulus_seq"] == 3

    async def test_delivery_ack_tool_result_is_a_stimulus(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """Every tool result is a stimulus (#1919): a ``signal_send`` delivery
        ack — even one carrying a legacy ``data['no_reaction']=true`` marker from
        before the removal — bumps ``last_stimulus_seq`` and leaves the session
        ACTIVE, so it wakes and the model gets a turn to react."""
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            # user (seq 1)
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "say hi to alice"},
            )
            # assistant reacts to the user (seq 2)
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "on it", "reacting_to": 1},
            )
            # delivery ack (seq 3) — a stimulus. The legacy ``no_reaction`` marker
            # is inert: ``is_stimulus`` no longer reads it.
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": "tc_send",
                    "content": '{"sent_at_ms": 1}',
                    "no_reaction": True,
                },
            )
            s = await _scalars(conn, session_id)
            session = await queries.get_session(conn, session_id, account_id=account_id)
        # last_stimulus_seq advances to 3 (the ack) — an unreacted stimulus.
        assert s["last_stimulus_seq"] == 3
        assert s["last_reacted_seq"] == 1
        assert session.status == "active"

    async def test_status_active_rescheduling_after_failed_step(
        self,
        pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A retry-pending session (user message, NO assistant reaction, then a
        rescheduling ``turn_ended``) stays ACTIVE so the sweep re-picks it.

        This is the harness/litellm error-retry path: the step fails before any
        assistant message is appended, ``_apply_retry_or_failure`` writes a
        rescheduling ``turn_ended``, and the unreacted user message must keep
        the session active. If ``turn_ended`` bumped ``last_reacted_seq``, this
        would flip idle and the retry would never fire (#749 over-correction).
        """
        pool, account_id, session_id = pool_and_session
        async with pool.acquire() as conn:
            # user message (seq 1) — never reacted to (the step failed)
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "hello"},
            )
            # rescheduling turn_ended (seq 2), no assistant reaction
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="lifecycle",
                data={
                    "event": "turn_ended",
                    "status": "rescheduling",
                    "stop_reason": "rescheduling",
                },
            )
            session = await queries.get_session(conn, session_id, account_id=account_id)
        # last_stimulus_seq=1 (user) > last_reacted_seq=0 → active.
        assert session.status == "active"

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
        — last_stimulus_seq > last_reacted_seq so the session is active."""
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
        # last_stimulus_seq=3 (tool result) > last_reacted_seq=2 (turn_ended)
        # AND open_tool_call_count went 1 -> 0 (but last_stimulus_seq > last_reacted_seq)
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
        assert clone_scalars["last_stimulus_seq"] == parent_scalars["last_stimulus_seq"]

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
