"""Regression suite for aios #841 — the ``open_tool_call_count`` leak on a
reused/duplicate ``tool_call_id``.

The bug
-------
``open_tool_call_count`` is a denormalized counter on the ``sessions`` row,
maintained transactionally inside ``append_event``. The assistant turn that
issues ``tool_calls`` increments the counter by ``len(tool_calls)`` — id-blind.
The matching -1 lands when the ``role:"tool"`` result append for that
``tool_call_id`` runs.

Both result-append paths dedup-guard on ``(session_id, tool_call_id)`` and
SHORT-CIRCUIT (no ``append_event``, so no -1) when a result already exists:

- ``harness.tool_dispatch._append_tool_result_event`` (worker)
- ``services.sessions.append_tool_result`` (API + connector runtime)

So when a ``tool_call_id`` is reused across two assistant turns, the second
turn's id-blind +1 has no matching -1 — the dedup-skip eats it. The counter
leaks a permanent +1, keeping ``open_tool_call_count > 0`` forever. The session
is then a permanent wake candidate (``_SESSION_ACTIVE_EXPR`` /
``CANDIDATE_ROWS_SQL``) and the sweep wakes it every ~30s for a full model call
with nothing to do.

The fix (#841)
--------------
A shared helper ``queries.decrement_open_tool_call_count`` applies the missing
-1 (``GREATEST(..., 0)``-clamped) at both dedup-skip sites, inside the existing
session-row-lock transaction.

The invariant this suite pins
-----------------------------
- A reused ``tool_call_id`` whose duplicate result append dedup-skips leaves
  ``open_tool_call_count`` at 0 (compensation fired), so the session settles
  idle and is NOT a wake candidate — via BOTH the worker path and the API path.
- The ``ConflictError`` branch (intent mismatch — success result then error
  result) is NOT a dedup-skip and must NOT compensate: the counter is unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.harness.sweep import find_sessions_needing_inference
from aios.harness.task_registry import TaskRegistry
from aios.harness.tool_dispatch import _append_tool_result_event
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


def _assistant(tool_call_ids: list[str], name: str = "bash") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tcid,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for tcid in tool_call_ids
        ],
    }


async def _open_count(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        count: int = await conn.fetchval(
            "SELECT open_tool_call_count FROM sessions WHERE id = $1", session_id
        )
        return count


async def _status(pool: asyncpg.Pool[Any], session_id: str, account_id: str) -> str:
    async with pool.acquire() as conn:
        session = await queries.get_session(conn, session_id, account_id=account_id)
    return session.status


async def _last_event_seq(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        seq: int = await conn.fetchval(
            "SELECT last_event_seq FROM sessions WHERE id = $1", session_id
        )
        return seq


@pytest.fixture
async def pool_account_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a fresh session whose
    agent declares ``bash`` (so the parent ``tool_calls`` entry resolves a
    tool name for the API result-append path)."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_dedup_leak"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "dedup-leak",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="dedup-leak",
            tools=[ToolSpec(type="bash")],
        )
        yield pool, account_id, session.id
    finally:
        await pool.close()


class TestWorkerDedupSkipCompensates:
    """The worker path (``_append_tool_result_event``) compensates the leak."""

    async def test_reused_tool_call_id_through_worker_settles_idle(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_account_session

        # 1. Assistant turn A1 issues tc_1 → id-blind +1 → count 1.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=_assistant(["tc_1"]),
            )

        # 2. Result for tc_1 lands → real -1 → count 0.
        await _append_tool_result_event(
            pool,
            session_id,
            "tc_1",
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok", "name": "bash"},
            account_id=account_id,
        )
        assert await _open_count(pool, session_id) == 0
        seq2 = await _last_event_seq(pool, session_id)

        # 3. Assistant turn A2 REUSES tc_1, reacting to the tool result so the
        #    stimulus is consumed → id-blind +1 → count 1 (the leak is armed).
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=_assistant(["tc_1"]) | {"reacting_to": seq2},
            )
        assert await _open_count(pool, session_id) == 1, "leak armed: id-blind +1 on reused id"

        # 4. Result for tc_1 AGAIN → dedup-skip → compensation must fire → count 0.
        await _append_tool_result_event(
            pool,
            session_id,
            "tc_1",
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok", "name": "bash"},
            account_id=account_id,
        )
        assert await _open_count(pool, session_id) == 0, (
            "dedup-skip on a reused tool_call_id must compensate the id-blind +1; "
            "without the fix the counter leaks +1 forever"
        )

        assert await _status(pool, session_id, account_id) == "idle"
        registry = TaskRegistry()
        candidates = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id not in candidates, (
            "with open_tool_call_count compensated to 0 and the stimulus reacted-to, "
            "the session must NOT be a wake candidate — the 30s wake loop is broken"
        )


class TestApiDedupSkipCompensates:
    """The API path (``services.append_tool_result``) compensates the leak."""

    async def test_reused_tool_call_id_through_api_settles_idle(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_account_session

        # 1. Assistant turn A1 issues tc_1 → +1 → count 1.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=_assistant(["tc_1"]),
            )

        # 2. Result for tc_1 via the API path → real -1 → count 0.
        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id="tc_1",
                content="ok",
            )
        assert await _open_count(pool, session_id) == 0
        seq2 = await _last_event_seq(pool, session_id)

        # 3. Assistant turn A2 REUSES tc_1, reacting to the tool result → +1 → count 1.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=_assistant(["tc_1"]) | {"reacting_to": seq2},
            )
        assert await _open_count(pool, session_id) == 1, "leak armed: id-blind +1 on reused id"

        # 4. Result for tc_1 AGAIN with the SAME is_error=False → idempotent
        #    dedup return → compensation must fire → count 0.
        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id="tc_1",
                content="ok",
            )
        assert await _open_count(pool, session_id) == 0, (
            "the idempotent dedup return must compensate the id-blind +1; "
            "without the fix the counter leaks +1 forever"
        )

        assert await _status(pool, session_id, account_id) == "idle"
        registry = TaskRegistry()
        candidates = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id not in candidates, (
            "with open_tool_call_count compensated to 0 and the stimulus reacted-to, "
            "the session must NOT be a wake candidate — the 30s wake loop is broken"
        )

    async def test_idempotent_retry_with_open_sibling_self_heals(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A genuine idempotent retry (not a reused id) compensates EVERY
        dedup-skip — including when a SIBLING call is still open, which
        transiently undercounts ``open_tool_call_count``. That undercount is
        SAFE and SELF-HEALING: ``GREATEST(..., 0)`` clamps the floor, and the
        independent ``last_stimulus_seq > last_reacted_seq`` wake path
        re-activates the session when the sibling's real result lands.
        """
        pool, account_id, session_id = pool_account_session
        registry = TaskRegistry()

        # 1. Assistant turn A1 issues tc_X and tc_Y → id-blind +2 → count 2.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=_assistant(["tc_X", "tc_Y"]),
            )
        assert await _open_count(pool, session_id) == 2

        # 2. Result for tc_X lands via the API path → real -1 → count 1.
        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id="tc_X",
                content="ok",
            )
        assert await _open_count(pool, session_id) == 1
        seq_x = await _last_event_seq(pool, session_id)

        # 3. Assistant turn A2 reacts to tc_X (no new tool_calls). tc_Y is still
        #    open, so the count stays 1.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "saw tc_X", "reacting_to": seq_x},
            )
        assert await _open_count(pool, session_id) == 1

        # 4. Spurious idempotent retry of tc_X (same is_error=False) → dedup-skip
        #    → compensation fires unconditionally → count 0, even though tc_Y is
        #    still genuinely open. This is the documented transient undercount.
        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id="tc_X",
                content="ok",
            )
        assert await _open_count(pool, session_id) == 0, (
            "the unconditional compensation fires on the idempotent retry even "
            "with a sibling (tc_Y) still open — the documented transient undercount"
        )

        # The transient false-idle: count==0 AND tc_X already reacted-to, so in
        # THIS window the session is not a wake candidate. This is the bounded,
        # self-healing window — tc_Y's real result (step 5) re-activates it.
        candidates = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id not in candidates

        # 5. tc_Y's REAL result lands → append bumps last_stimulus_seq.
        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id="tc_Y",
                content="ok",
            )
        seq_y = await _last_event_seq(pool, session_id)

        # SELF-HEALING: the stimulus path re-woke the session even though
        # open_tool_call_count is clamped at 0. This is the key assertion that
        # proves the unconditional-compensation design is safe.
        candidates = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in candidates, (
            "tc_Y's real result bumped last_stimulus_seq, re-activating the "
            "session via the stimulus path despite the clamped counter — the "
            "transient undercount is self-healing, never a permanent stall"
        )

        # 6. Assistant turn A3 reacts to tc_Y → final settle: idle, not a candidate.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "assistant", "content": "saw tc_Y", "reacting_to": seq_y},
            )
        assert await _status(pool, session_id, account_id) == "idle"
        candidates = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id not in candidates

    async def test_intent_mismatch_conflict_does_not_decrement(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """The ``ConflictError`` branch (success result then ERROR result for
        the same id) is a genuine conflict, NOT a dedup-skip — no compensation
        is owed, so the counter must be unchanged."""
        pool, account_id, session_id = pool_account_session

        # Assistant turn issues tc_1 → +1 → count 1.
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=_assistant(["tc_1"]),
            )

        # Success result resolves tc_1 → real -1 → count 0.
        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id="tc_1",
                content="ok",
                is_error=False,
            )
        assert await _open_count(pool, session_id) == 0

        # An ERROR result for the SAME id is an intent mismatch → ConflictError.
        with pytest.raises(ConflictError):
            async with pool.acquire() as conn:
                await sessions_service.append_tool_result(
                    conn,
                    account_id=account_id,
                    session_id=session_id,
                    tool_call_id="tc_1",
                    content="boom",
                    is_error=True,
                )

        # The conflict path must NOT compensate — the counter is unchanged.
        assert await _open_count(pool, session_id) == 0, (
            "the ConflictError branch is a genuine conflict, not a dedup-skip; "
            "it must not touch open_tool_call_count"
        )
