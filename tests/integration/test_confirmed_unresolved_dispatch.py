"""Integration test for ``queries.list_confirmed_unresolved_tool_calls`` — the
unwindowed dispatch-side resolver introduced for #737 (and unified with the
sweep's case-(c) wake predicate, ``sweep.CONFIRMED_ROWS_SQL``).

Exercises the three filters against real Postgres (so the JSONB ``@>`` join,
the ``NOT EXISTS`` result guard, and the ``events_tool_confirmed_allow_idx``
partial index from migration 0065 are all genuinely in play):

* confirmed (allow) ∧ no result ∧ parent is a NON-latest assistant  → returned
  (the #737 window-edge / non-latest-assistant case the resolver must recover);
* confirmed (allow) ∧ a tool_result already exists                  → excluded
  (no re-dispatch — CLAUDE.md invariant #4);
* an unconfirmed tool_call (no ``tool_confirmed allow``)            → excluded.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.agents import ToolSpec
from aios.models.events import EventKind
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


def _allow(tool_call_id: str) -> dict[str, Any]:
    return {"event": "tool_confirmed", "result": "allow", "tool_call_id": tool_call_id}


@pytest.fixture
async def session_with_mixed_confirmations(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a session whose log is:

    A1[tc_pending, tc_unconfirmed] → user → A2[tc_resolved],
    then allow(tc_pending), allow(tc_resolved), tool_result(tc_resolved).
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_confirmed_unresolved_dispatch"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "confirmed-unresolved-dispatch-test",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="confirmed-unresolved-dispatch",
            tools=[ToolSpec(type="bash")],
        )
        sid = session.id

        async def append(kind: EventKind, data: dict[str, Any]) -> None:
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn, account_id=account_id, session_id=sid, kind=kind, data=data
                )

        # A1 carries the pending tool (later confirmed, never resolved) plus an
        # unconfirmed tool; A2 is the LATEST assistant and carries the resolved
        # tool — so the pending tool's parent is deliberately not the latest.
        await append("message", _assistant(["tc_pending", "tc_unconfirmed"]))
        await append("message", {"role": "user", "content": "are you still there?"})
        await append("message", _assistant(["tc_resolved"]))
        await append("lifecycle", _allow("tc_pending"))
        await append("lifecycle", _allow("tc_resolved"))
        await append(
            "message",
            {
                "role": "tool",
                "tool_call_id": "tc_resolved",
                "content": "done",
                "name": "bash",
            },
        )
        yield pool, account_id, sid
    finally:
        await pool.close()


class TestListConfirmedUnresolvedToolCalls:
    async def test_only_confirmed_unresolved_are_returned(
        self,
        session_with_mixed_confirmations: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = session_with_mixed_confirmations
        async with pool.acquire() as conn:
            dispatchable = await queries.list_confirmed_unresolved_tool_calls(
                conn, session_id, account_id=account_id
            )

        ids = [tc["id"] for tc in dispatchable]
        assert ids == ["tc_pending"], (
            "expected only the confirmed-and-unresolved tool_call (on the "
            "non-latest assistant A1); got "
            f"{ids} — tc_resolved has a result (invariant #4) and tc_unconfirmed "
            "was never allowed"
        )
        # Full dispatchable dict, ready for ``launch_tool_calls``.
        assert dispatchable[0]["function"]["name"] == "bash"

    async def test_account_scoped(
        self,
        session_with_mixed_confirmations: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A different account sees nothing for the same session id (tenant
        isolation — the resolver filters ``account_id``)."""
        pool, _account_id, session_id = session_with_mixed_confirmations
        async with pool.acquire() as conn:
            dispatchable = await queries.list_confirmed_unresolved_tool_calls(
                conn, session_id, account_id="acc_someone_else"
            )
        assert dispatchable == []
