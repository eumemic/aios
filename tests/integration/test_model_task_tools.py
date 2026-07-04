"""Integration tests for the model-facing task verbs — ``stop_task`` + ``list_tasks`` (#1428).

DB-backed (testcontainer Postgres). The model plane keys on ``tool_call_id`` (the handle the
caller already holds, stamped on the servicer edge by ``invoke_session._caller``); these tests
exercise:

* ``list_open_tasks`` (backs ``list_tasks``) — only OPEN edges keyed by ``tool_call_id``
  appear; an answered one drops off, and another session's edge is never visible. Both servicer
  kinds (a session servicer + a run servicer).
* ``stop_task`` — seeds the cancel on the servicer (session arm → cancel-marker; run arm →
  cancel signal, the launcher guard satisfied by construction since ``find_parked_servicer`` pins
  ``caller.id``), reports "already resolved" on a terminal edge, and errors on a foreign/absent
  ``tool_call_id``.
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
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.models.agents import ToolSpec
from aios.models.sessions import Ok
from aios.services import tasks as tasks_service
from aios.services import workflows as wf_service
from aios.tools import tasks as task_tools
from aios.tools import workflow_completion
from aios.tools.registry import ToolResult
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_tasks"
_FROZEN: dict[str, list[Any]] = {"tools": [], "mcp_servers": [], "http_servers": []}


@pytest.fixture
async def env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, account_id)`` with the worker runtime wired so the task handlers run
    without a live worker (the wake/cancel-wake deferrals are patched out)."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev_pool, prev_reg = runtime.pool, runtime.inflight_tool_registry
    runtime.pool = pool
    runtime.inflight_tool_registry = InflightToolRegistry()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'tasks-root')",
                _ACCOUNT,
            )
        with (
            mock.patch("aios.services.sessions.defer_wake", new=AsyncMock()),
            mock.patch("aios.services.tasks.defer_wake", new=AsyncMock()),
            mock.patch("aios.jobs.app.defer_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
        ):
            yield pool, _ACCOUNT
    finally:
        runtime.pool, runtime.inflight_tool_registry = prev_pool, prev_reg
        await pool.close()


async def _seed_session(pool: asyncpg.Pool[Any], prefix: str) -> str:
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=_ACCOUNT, prefix=prefix, tools=[ToolSpec(type="bash")]
    )
    return session.id


async def _write_session_edge(
    pool: asyncpg.Pool[Any],
    *,
    servicer_id: str,
    caller_session_id: str,
    tool_call_id: str,
    request_id: str,
) -> None:
    """Write the servicer's ``request_opened`` edge carrying the caller's tool_call_id — exactly
    what a ``call_session``/``call_agent`` handler stamps via ``_caller``."""
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=servicer_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            caller={"kind": "session", "id": caller_session_id, "tool_call_id": tool_call_id},
            depth=1,
            environment_id="env_x",
            frozen_surface=_FROZEN,
            vault_ids=[],
            output_schema=None,
        )


async def _seed_run(
    pool: asyncpg.Pool[Any],
    *,
    caller_session_id: str,
    tool_call_id: str,
    request_id: str,
) -> str:
    """Insert a pending run servicer whose caller edge names ``caller_session_id`` (the launcher)
    and carries ``tool_call_id`` — what ``launch_awaited_run`` writes for a ``call_workflow``."""
    wf = await wf_service.create_workflow(
        pool,
        account_id=_ACCOUNT,
        name=f"wf-{tool_call_id}",
        script="async def main(input):\n    return None\n",
        description=None,
        tools=[],
    )
    _agent, environment, _sess = await seed_agent_env_session(
        pool, account_id=_ACCOUNT, prefix=f"runenv-{tool_call_id}"
    )
    async with pool.acquire() as conn:
        run = await wf_queries.insert_wf_run(
            conn,
            account_id=_ACCOUNT,
            workflow_id=wf.id,
            environment_id=environment.id,
            parent_run_id=None,
            launcher_session_id=caller_session_id,
            request_id=request_id,
            caller={
                "kind": "session",
                "id": caller_session_id,
                "tool_call_id": tool_call_id,
                "awaited": True,
            },
            script=wf.script,
            script_sha="x" * 64,
            host_semantics_epoch=1,
            input=None,
            tools=[],
            mcp_servers=[],
            http_servers=[],
            budget_usd=None,
            default_child_model="openrouter/test",
            depth=10,
        )
    return run.id


# ─── list_open_tasks (backs list_tasks) ────────────────────────────────


async def test_list_open_tasks_only_open_and_own(
    env: tuple[asyncpg.Pool[Any], str],
) -> None:
    """Only this session's OPEN edges appear, keyed by tool_call_id — an answered edge and a
    foreign session's edge are both excluded; both servicer kinds are listed."""
    pool, account_id = env
    caller = await _seed_session(pool, "list-caller")

    # tc_open: a session servicer, still pending → listed.
    servicer_open = await _seed_session(pool, "list-srv-open")
    await _write_session_edge(
        pool,
        servicer_id=servicer_open,
        caller_session_id=caller,
        tool_call_id="tc_open",
        request_id="req_open",
    )
    # tc_answered: a session servicer that already answered → dropped.
    servicer_done = await _seed_session(pool, "list-srv-done")
    await _write_session_edge(
        pool,
        servicer_id=servicer_done,
        caller_session_id=caller,
        tool_call_id="tc_answered",
        request_id="req_answered",
    )
    await workflow_completion.respond_to_request(
        pool, servicer_done, request_id="req_answered", outcome=Ok(result={"v": 1})
    )
    # tc_run: a run servicer, pending → listed.
    run_id = await _seed_run(
        pool, caller_session_id=caller, tool_call_id="tc_run", request_id="req_run"
    )
    # tc_foreign: another session's edge → never visible to `caller`.
    other = await _seed_session(pool, "list-other")
    servicer_foreign = await _seed_session(pool, "list-srv-foreign")
    await _write_session_edge(
        pool,
        servicer_id=servicer_foreign,
        caller_session_id=other,
        tool_call_id="tc_foreign",
        request_id="req_foreign",
    )

    open_tasks = await tasks_service.list_open_tasks(pool, session_id=caller, account_id=account_id)
    by_tcid = {i.tool_call_id: i for i in open_tasks}
    assert set(by_tcid) == {"tc_open", "tc_run"}  # answered + foreign excluded
    assert (by_tcid["tc_open"].kind, by_tcid["tc_open"].target) == ("session", servicer_open)
    assert (by_tcid["tc_run"].kind, by_tcid["tc_run"].target) == ("run", run_id)


async def test_list_tasks_handler_shape(env: tuple[asyncpg.Pool[Any], str]) -> None:
    """The list_tasks tool returns the open roster as a JSON ``{tasks: [...]}`` envelope."""
    pool, _account_id = env
    caller = await _seed_session(pool, "lt-caller")
    servicer = await _seed_session(pool, "lt-srv")
    await _write_session_edge(
        pool,
        servicer_id=servicer,
        caller_session_id=caller,
        tool_call_id="tc_a",
        request_id="req_a",
    )

    out = await task_tools.list_tasks_handler(caller, {})
    assert list(out) == ["tasks"]
    assert len(out["tasks"]) == 1
    entry = out["tasks"][0]
    assert entry["tool_call_id"] == "tc_a"
    assert entry["kind"] == "session"
    assert entry["target"] == servicer
    assert "opened_at" in entry


# ─── stop_task ───────────────────────────────────────────────────────────────


async def test_stop_task_session_arm_seeds_cancel_marker(
    env: tuple[asyncpg.Pool[Any], str],
) -> None:
    """stop_task on a session servicer seeds the cancel-marker the target's step harvests."""
    pool, _account_id = env
    caller = await _seed_session(pool, "stop-caller")
    servicer = await _seed_session(pool, "stop-srv")
    await _write_session_edge(
        pool,
        servicer_id=servicer,
        caller_session_id=caller,
        tool_call_id="tc_stop",
        request_id="req_stop",
    )

    out = await task_tools.stop_task_handler(caller, {"tool_call_id": "tc_stop"})
    assert out == {"ok": "stop requested"}

    async with pool.acquire() as conn:
        marker = await conn.fetchrow(
            "SELECT 1 FROM session_cancel_markers WHERE session_id = $1 AND request_id = $2",
            servicer,
            "req_stop",
        )
    assert marker is not None


async def test_stop_task_run_arm_seeds_cancel_signal(
    env: tuple[asyncpg.Pool[Any], str],
) -> None:
    """stop_task on a run servicer seeds the cancel signal — the launcher guard is satisfied by
    construction (find_parked_servicer pins caller.id = the launching session)."""
    pool, _account_id = env
    caller = await _seed_session(pool, "stop-run-caller")
    run_id = await _seed_run(
        pool, caller_session_id=caller, tool_call_id="tc_run_stop", request_id="req_run_stop"
    )

    out = await task_tools.stop_task_handler(caller, {"tool_call_id": "tc_run_stop"})
    assert out == {"ok": "stop requested"}

    async with pool.acquire() as conn:
        signal = await conn.fetchrow(
            "SELECT 1 FROM wf_run_signals WHERE run_id = $1 AND kind = 'cancel'", run_id
        )
    assert signal is not None


async def test_stop_task_foreign_tool_call_id_errors(
    env: tuple[asyncpg.Pool[Any], str],
) -> None:
    """An absent / foreign tool_call_id resolves to None (caller.id pinned) → a clean
    model-visible error, never another session's task and no cancel seeded."""
    pool, _account_id = env
    caller = await _seed_session(pool, "stop-foreign-caller")
    other = await _seed_session(pool, "stop-foreign-other")
    servicer = await _seed_session(pool, "stop-foreign-srv")
    # An edge owned by `other`, not `caller`.
    await _write_session_edge(
        pool,
        servicer_id=servicer,
        caller_session_id=other,
        tool_call_id="tc_otherown",
        request_id="req_otherown",
    )

    # caller cannot reach other's task by its tool_call_id.
    out = await task_tools.stop_task_handler(caller, {"tool_call_id": "tc_otherown"})
    assert isinstance(out, ToolResult)
    assert out.is_error is True
    assert isinstance(out.content, str) and "no open task" in out.content

    # No cancel-marker leaked onto the foreign servicer.
    async with pool.acquire() as conn:
        leaked = await conn.fetchrow(
            "SELECT 1 FROM session_cancel_markers WHERE session_id = $1", servicer
        )
    assert leaked is None


async def test_stop_task_already_resolved(env: tuple[asyncpg.Pool[Any], str]) -> None:
    """A task that already answered reports 'already resolved' and seeds NO cancel."""
    pool, _account_id = env
    caller = await _seed_session(pool, "stop-done-caller")
    servicer = await _seed_session(pool, "stop-done-srv")
    await _write_session_edge(
        pool,
        servicer_id=servicer,
        caller_session_id=caller,
        tool_call_id="tc_done",
        request_id="req_done",
    )
    await workflow_completion.respond_to_request(
        pool, servicer, request_id="req_done", outcome=Ok(result={"v": 9})
    )

    out = await task_tools.stop_task_handler(caller, {"tool_call_id": "tc_done"})
    assert out == {"ok": "already resolved"}

    async with pool.acquire() as conn:
        marker = await conn.fetchrow(
            "SELECT 1 FROM session_cancel_markers WHERE session_id = $1 AND request_id = $2",
            servicer,
            "req_done",
        )
    assert marker is None  # already terminal → nothing seeded
