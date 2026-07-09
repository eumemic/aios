"""Integration coverage for the #1756 confirm-then-interrupt dispatch guard.

``_dispatch_confirmed_tools`` (``harness/loop.py``) resolves each
``tool_confirmed``/``allow`` call against the session's latest ``interrupt``
event seq before cold-dispatching: a call confirmed BEFORE the interrupt must
not fire (it is resolved in-place as ``cancelled``), while a FRESH
confirmation issued AFTER the interrupt still dispatches (the #746 "fresh
confirm of an old proposal is fresh intent" rule, applied at the interrupt
boundary).

Exercised end-to-end against real Postgres: the durable ``interrupt`` event
written by ``POST /sessions/:id/interrupt`` (here inserted directly, mirroring
``routers/sessions.py``), the new ``queries.find_latest_interrupt_seq`` /
``queries.find_tool_confirmed_seqs`` resolvers, and ``_dispatch_confirmed_tools``
itself — including the in-place cancellation write.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.loop import _dispatch_confirmed_tools
from aios.models.agents import ToolSpec
from aios.models.events import EventKind
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _inflight_tool_registry() -> Any:
    """``resolve_confirmed_call_as_cancelled``'s tail sweep
    (``tool_dispatch._trigger_sweep``) reads
    ``runtime.require_inflight_tool_registry()`` — only ever populated inside
    a running worker. Stub it for this module's direct calls into the
    dispatch guard, mirroring the existing convention in
    ``tests/integration/test_wf_step.py``.
    """
    prev = runtime.inflight_tool_registry
    runtime.inflight_tool_registry = InflightToolRegistry()
    try:
        yield runtime.inflight_tool_registry
    finally:
        runtime.inflight_tool_registry = prev


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


def _allow(tool_call_id: str) -> dict[str, Any]:
    return {"event": "tool_confirmed", "result": "allow", "tool_call_id": tool_call_id}


@pytest.fixture
async def session_with_confirm_then_interrupt(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a session whose log is:

    A1[tc_stale, tc_fresh] → allow(tc_stale) → interrupt → allow(tc_fresh).

    ``tc_stale`` was confirmed BEFORE the interrupt; ``tc_fresh`` is confirmed
    AFTER it — the two cases the guard must partition.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_confirm_then_interrupt"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "confirm-then-interrupt-test",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="confirm-then-interrupt",
            tools=[ToolSpec(type="bash")],
        )
        sid = session.id

        async def append(kind: EventKind, data: dict[str, Any]) -> None:
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn, account_id=account_id, session_id=sid, kind=kind, data=data
                )

        await append("message", _assistant(["tc_stale", "tc_fresh"]))
        await append("lifecycle", _allow("tc_stale"))
        await append("interrupt", {"reason": "operator changed their mind"})
        await append("lifecycle", _allow("tc_fresh"))

        yield pool, account_id, sid
    finally:
        await pool.close()


class TestFindLatestInterruptSeq:
    async def test_returns_seq_of_latest_interrupt(
        self,
        session_with_confirm_then_interrupt: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = session_with_confirm_then_interrupt
        async with pool.acquire() as conn:
            interrupt_seq = await queries.find_latest_interrupt_seq(
                conn, session_id, account_id=account_id
            )
            stale_seq, fresh_seq = (
                r["seq"]
                for r in await conn.fetch(
                    "SELECT seq FROM events WHERE session_id = $1 "
                    "AND data->>'tool_call_id' = ANY($2::text[]) "
                    "ORDER BY seq ASC",
                    session_id,
                    ["tc_stale", "tc_fresh"],
                )
            )
        assert interrupt_seq is not None
        assert stale_seq < interrupt_seq < fresh_seq

    async def test_returns_none_when_never_interrupted(
        self, migrated_db_url: str, _reset_db_state: None
    ) -> None:
        pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
        try:
            account_id = "acc_never_interrupted"
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO accounts "
                    "(id, parent_account_id, can_mint_children, display_name) "
                    "VALUES ($1, NULL, TRUE, $2)",
                    account_id,
                    "never-interrupted-test",
                )
            _agent, _env, session = await seed_agent_env_session(
                pool, account_id=account_id, prefix="never-interrupted"
            )
            async with pool.acquire() as conn:
                interrupt_seq = await queries.find_latest_interrupt_seq(
                    conn, session.id, account_id=account_id
                )
            assert interrupt_seq is None
        finally:
            await pool.close()


class TestDispatchConfirmedToolsInterruptGuard:
    async def test_stale_confirm_cancelled_fresh_confirm_dispatched(
        self,
        session_with_confirm_then_interrupt: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """The call confirmed BEFORE the interrupt is resolved in-place as
        ``cancelled`` (not returned for dispatch); the one confirmed AFTER it
        dispatches normally."""
        pool, account_id, session_id = session_with_confirm_then_interrupt

        # resolve_confirmed_call_as_cancelled's tail sweep
        # (tool_dispatch._trigger_sweep) is production behaviour for an
        # in-flight task's finally block, not for this synchronous
        # dispatch-resolution call — the normal caller (the harness step)
        # marks a returned call in-flight (and then launches it) before any
        # other sweep can observe it as "confirmed, unresolved, not
        # in-flight" and misclassify it as lost (sweep.py case (c)). Stub the
        # tail sweep out here so this test asserts the dispatch guard's own
        # partitioning in isolation, without racing that unrelated sweep path.
        with mock.patch("aios.harness.tool_dispatch._trigger_sweep", new=AsyncMock()):
            pending = await _dispatch_confirmed_tools(
                pool,
                session_id,
                account_id=account_id,
                inflight_tool_registry=InflightToolRegistry(),
            )

        assert [tc["id"] for tc in pending] == ["tc_fresh"], (
            "expected only the post-interrupt fresh confirmation to be "
            f"returned for dispatch; got {[tc['id'] for tc in pending]}"
        )

        # tc_stale must have been resolved in-place as a cancelled tool_result
        # — not left dangling for the sweep to keep re-waking on.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM events WHERE session_id = $1 AND kind = 'message' "
                "AND role = 'tool' AND data->>'tool_call_id' = $2",
                session_id,
                "tc_stale",
            )
        assert row is not None, "tc_stale must have a tool_result event after dispatch"
        assert row["data"]["is_error"] is True
        assert row["data"]["content"] == '{"error": "cancelled"}'

        # tc_fresh must NOT have a result yet — it was returned for the
        # caller (harness step) to actually launch, not synchronously resolved.
        async with pool.acquire() as conn:
            fresh_rows = await conn.fetch(
                "SELECT kind, role, data::text FROM events WHERE session_id = $1 "
                "AND data->>'tool_call_id' = $2 ORDER BY seq",
                session_id,
                "tc_fresh",
            )
        result_rows = [r for r in fresh_rows if r["kind"] == "message" and r["role"] == "tool"]
        assert not result_rows, f"tc_fresh unexpectedly has a tool_result: {fresh_rows}"

    async def test_no_interrupt_dispatches_everything(
        self, migrated_db_url: str, _reset_db_state: None
    ) -> None:
        """No interrupt on the session: every confirmed-unresolved call
        dispatches (the guard is a pure no-op absent an interrupt event)."""
        pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
        try:
            account_id = "acc_no_interrupt_dispatch"
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO accounts "
                    "(id, parent_account_id, can_mint_children, display_name) "
                    "VALUES ($1, NULL, TRUE, $2)",
                    account_id,
                    "no-interrupt-dispatch-test",
                )
            _agent, _env, session = await seed_agent_env_session(
                pool,
                account_id=account_id,
                prefix="no-interrupt-dispatch",
                tools=[ToolSpec(type="bash")],
            )
            sid = session.id

            async def append(kind: EventKind, data: dict[str, Any]) -> None:
                async with pool.acquire() as conn:
                    await queries.append_event(
                        conn, account_id=account_id, session_id=sid, kind=kind, data=data
                    )

            await append("message", _assistant(["tc_only"]))
            await append("lifecycle", _allow("tc_only"))

            pending = await _dispatch_confirmed_tools(
                pool,
                sid,
                account_id=account_id,
                inflight_tool_registry=InflightToolRegistry(),
            )
            assert [tc["id"] for tc in pending] == ["tc_only"]
        finally:
            await pool.close()
