"""Integration test: a worker's late tool_result append for an in-flight
tool whose deny already committed must NOT silently overwrite the deny.

Scenario (the dual race to #535):

  1. Model emits ``tool_call X`` as an ``always_allow`` builtin (or any
     tool the harness dispatches without operator confirmation).
  2. Worker fire-and-forgets the tool task; the tool starts running.
  3. Operator POSTs ``decision=deny`` for ``X`` via
     ``/v1/sessions/<id>/tool-confirmations``. The deny path appends a
     ``role:"tool"`` event with ``is_error=True``.
  4. Tool task completes a moment later. The worker's success-result
     append (``aios.harness.tool_dispatch._execute_tool_async``) calls
     ``sessions_service.append_event`` directly — no
     ``tool_call_id`` dedup. A SECOND ``role:"tool"`` event for the
     SAME ``tool_call_id`` lands.

Effect at the model: the context builder
(``aios.harness.context._build_messages``) keys ``real_results`` by
``tool_call_id`` last-wins; the worker's late success event clobbers the
operator's deny, and the next prompt carries the tool's successful
output as if no deny had happened. Symmetric silent-failure to the bug
PR #535 closed in the opposite direction.

Per the same intent-mismatch posture #535 introduced for the API path,
the worker's append should refuse when an opposite-intent tool-role
event already exists. Same dedup-by-``tool_call_id`` machinery, just
moved one layer up so BOTH commit paths (operator + worker) honor it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.tool_dispatch import _execute_tool_async
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from aios.tools.registry import registry
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def runtime_for_dispatch() -> AsyncIterator[None]:
    """Install minimum worker globals on ``aios.harness.runtime`` so
    ``_execute_tool_async``'s ``runtime.require_inflight_tool_registry()`` works.

    Restores the previous globals on teardown.
    """
    prev_inflight_reg = runtime.inflight_tool_registry
    prev_worker_id = runtime.worker_id
    runtime.inflight_tool_registry = InflightToolRegistry()
    runtime.worker_id = "worker_test_deny_race"
    try:
        yield
    finally:
        runtime.inflight_tool_registry = prev_inflight_reg
        runtime.worker_id = prev_worker_id


@pytest.fixture
def _register_test_tool() -> Any:
    """Snapshot/restore the tool registry around a test-local tool.

    Registers an ``echo_for_deny_race`` tool whose handler immediately
    returns success. Drives ``_execute_tool_async`` end-to-end without
    the bash-tool side effects.
    """
    snapshot = dict(registry._tools)

    async def _handler(_session_id: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        return {"output": "tool actually ran successfully"}

    registry.register(
        name="echo_for_deny_race",
        description="test-only tool that returns success immediately",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_handler,
    )
    try:
        yield
    finally:
        registry._tools = snapshot


@pytest.fixture
async def session_with_pending_tool_call(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, session_id, tool_call_id)`` for a
    session whose event log contains an assistant message with a
    tool_calls entry but NO tool-role result event yet (in-flight)."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_worker_race', NULL, TRUE, 'worker-deny-race-test')
                """
            )
        # ``bash`` is a placeholder — the test invokes
        # ``_execute_tool_async`` directly with a tool_call whose
        # function.name targets the test-only registered tool, so
        # the agent's declared tools list is not consulted.
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id="acc_worker_race",
            prefix="worker-deny-race",
            tools=[ToolSpec(type="bash")],
        )
        async with pool.acquire() as conn:
            # Assistant message with one tool_call; no tool-role result
            # event yet (the tool is in-flight in the simulated race).
            await queries.append_event(
                conn,
                account_id="acc_worker_race",
                session_id=session.id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_in_flight",
                            "type": "function",
                            "function": {
                                "name": "echo_for_deny_race",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
            )
        yield pool, "acc_worker_race", session.id, "tc_in_flight"
    finally:
        await pool.close()


class TestWorkerResultAfterDeny:
    async def test_worker_success_after_deny_must_not_clobber_deny(
        self,
        session_with_pending_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
        runtime_for_dispatch: None,
        _register_test_tool: None,
    ) -> None:
        """Race: operator deny commits while tool task is in-flight; the
        task's late success append must not land a second tool-role
        event that the context builder's last-wins keying would let
        clobber the deny.

        Concrete assertions:
          * exactly ONE ``role:"tool"`` event exists for the tool_call_id
            after both commit attempts
          * that single event is the operator's deny (``is_error == True``)
        """
        pool, account_id, session_id, tool_call_id = session_with_pending_tool_call

        # Step A: operator deny lands. Today this writes a tool-role
        # event with is_error=True via append_tool_result's dedup-aware
        # path.
        await sessions_service.confirm_tool_deny(
            pool,
            session_id,
            tool_call_id,
            "operator denied while task was in-flight",
            account_id=account_id,
        )

        # Step B: worker's tool task completes and tries to log its
        # success. Drive the real production code path (``_execute_tool_async``)
        # directly — the task body is the SQL-commit surface that
        # races the operator deny in production. Awaiting it inline
        # eliminates the asyncio-scheduling variable; the race
        # ordering (deny first, worker second) is established by the
        # explicit step order.
        #
        # ``_trigger_sweep`` (called from the lifecycle's ``finally``)
        # depends on the procrastinate app being open; the SQL-commit
        # surface under test happens BEFORE the sweep so a no-op
        # patch is sufficient to keep the dispatch path intact.
        async def _noop_sweep(*_a: Any, **_kw: Any) -> None:
            return None

        with mock.patch("aios.harness.tool_dispatch._trigger_sweep", _noop_sweep):
            await _execute_tool_async(
                pool,
                session_id,
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "echo_for_deny_race", "arguments": "{}"},
                },
                account_id=account_id,
            )

        # The bug: the deny silently lost. Two tool-role events exist
        # for the same tool_call_id; the context builder's last-wins
        # keying renders the worker's success in the prompt.
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM events
                 WHERE session_id = $1
                   AND data->>'role' = 'tool'
                   AND data->>'tool_call_id' = $2
                 ORDER BY seq
                """,
                session_id,
                tool_call_id,
            )

        assert len(rows) == 1, (
            f"deny must be the only tool-role event; the worker's late append "
            f"clobbered the operator's intent (count={len(rows)})"
        )
        surviving = (
            json.loads(rows[0]["data"]) if isinstance(rows[0]["data"], str) else rows[0]["data"]
        )
        assert surviving.get("is_error") is True, (
            "the deny event must remain the authoritative tool-role event"
        )
