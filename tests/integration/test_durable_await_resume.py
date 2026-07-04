"""Integration tests for durable await-resume of parked ``call_*`` tasks (#1431).

DB-backed (testcontainer Postgres). On a worker crash a parked ``call_*`` tool task is
lost; recovery must re-derive its servicer from the durable edge (the ``tool_call_id`` the
handler stamped onto ``caller``) and RE-PARK it — re-attaching the servicer's exactly-once
answer to the original tool result — rather than orphaning it behind a synthetic error.

Covers:

* ``find_parked_servicer`` — the locator: resolves a session servicer (with its
  ``request_id`` + ``output_schema``) and a run servicer from the caller edge, and ``None``
  when no edge matches (the launch crashed before it was durable).
* ``find_and_repair_ghosts`` routing — a resumable ``call_*`` ghost with a live edge is
  RE-PARKED (not error-repaired); one with no edge falls through to a retryable
  ``launch_lost`` error; a non-``call_*`` ghost keeps the existing error-repair.
* the resume mechanism — ``_resume_parked_async`` lands the servicer's answer as the
  original tool result and writes NO ``tool_execute_start`` span (the span-less bracket, so
  a second-crash classification can't be misled into "may have completed").
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
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
from aios.harness.sweep import find_and_repair_ghosts
from aios.models.agents import ToolSpec
from aios.models.sessions import Ok
from aios.services import workflows as wf_service
from aios.tools import workflow_completion
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_resume"
_FROZEN: dict[str, list[Any]] = {"tools": [], "mcp_servers": [], "http_servers": []}


@pytest.fixture
async def env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, migrated_db_url)`` with the worker runtime wired so
    recovery + resume can run without a live worker (wakes are patched out)."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev_pool, prev_reg = runtime.pool, runtime.inflight_tool_registry
    runtime.pool = pool
    runtime.inflight_tool_registry = InflightToolRegistry()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'resume-root')",
                _ACCOUNT,
            )
        with (
            mock.patch("aios.services.sessions.defer_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
        ):
            yield pool, _ACCOUNT, migrated_db_url
    finally:
        runtime.pool, runtime.inflight_tool_registry = prev_pool, prev_reg
        await pool.close()


def _assistant(tool_calls: list[tuple[str, str]]) -> dict[str, Any]:
    """An assistant turn issuing ``(tool_call_id, tool_name)`` calls, no results."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": tcid, "type": "function", "function": {"name": name, "arguments": "{}"}}
            for tcid, name in tool_calls
        ],
    }


async def _seed_session(pool: asyncpg.Pool[Any], prefix: str, *, tools: list[ToolSpec]) -> str:
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=_ACCOUNT, prefix=prefix, tools=tools
    )
    return session.id


async def _write_session_edge(
    pool: asyncpg.Pool[Any],
    *,
    servicer_id: str,
    caller_session_id: str,
    tool_call_id: str,
    request_id: str,
    output_schema: dict[str, Any] | None = None,
) -> None:
    """Write the servicer's ``request_opened`` edge carrying the caller's tool_call_id —
    exactly what a ``call_session``/``call_agent`` handler stamps via ``_caller``."""
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
            output_schema=output_schema,
        )


async def _tool_results(
    pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'message' "
            "AND role = 'tool' AND data->>'tool_call_id' = $2",
            session_id,
            tool_call_id,
        )
    return [r["data"] for r in rows]


# ─── find_parked_servicer (the locator) ──────────────────────────────────────


async def test_find_parked_servicer_session(env: tuple[asyncpg.Pool[Any], str, str]) -> None:
    pool, account_id, _ = env
    caller = await _seed_session(pool, "loc-caller", tools=[ToolSpec(type="call_session")])
    servicer = await _seed_session(pool, "loc-servicer", tools=[ToolSpec(type="bash")])
    schema = {"type": "object"}
    await _write_session_edge(
        pool,
        servicer_id=servicer,
        caller_session_id=caller,
        tool_call_id="tc_sess",
        request_id="req_sess",
        output_schema=schema,
    )

    async with pool.acquire() as conn:
        handle = await queries.find_parked_servicer(
            conn, caller_session_id=caller, tool_call_id="tc_sess", account_id=account_id
        )
    assert handle == ("session", servicer, "req_sess", schema)


async def test_find_parked_servicer_run(env: tuple[asyncpg.Pool[Any], str, str]) -> None:
    pool, account_id, _ = env
    _agent, environment, caller_session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="loc-run-caller", tools=[ToolSpec(type="call_workflow")]
    )
    caller = caller_session.id
    wf = await wf_service.create_workflow(
        pool,
        account_id=account_id,
        name="resume-loc-wf",
        script="async def main(input):\n    return None\n",
        description=None,
        tools=[],
    )
    async with pool.acquire() as conn:
        run = await wf_queries.insert_wf_run(
            conn,
            account_id=account_id,
            workflow_id=wf.id,
            environment_id=environment.id,
            parent_run_id=None,
            launcher_session_id=caller,
            request_id="req_run",
            caller={"kind": "session", "id": caller, "tool_call_id": "tc_run"},
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

    async with pool.acquire() as conn:
        handle = await queries.find_parked_servicer(
            conn, caller_session_id=caller, tool_call_id="tc_run", account_id=account_id
        )
    # A run parks on its terminal row, so request_id is None for the park handle.
    assert handle == ("run", run.id, None, None)


async def test_find_parked_servicer_absent(env: tuple[asyncpg.Pool[Any], str, str]) -> None:
    pool, account_id, _ = env
    caller = await _seed_session(pool, "loc-absent", tools=[ToolSpec(type="call_session")])
    servicer = await _seed_session(pool, "loc-absent-srv", tools=[ToolSpec(type="bash")])
    await _write_session_edge(
        pool,
        servicer_id=servicer,
        caller_session_id=caller,
        tool_call_id="tc_real",
        request_id="req_real",
    )
    async with pool.acquire() as conn:
        # A different tool_call_id (and a different caller) must not match.
        assert (
            await queries.find_parked_servicer(
                conn, caller_session_id=caller, tool_call_id="tc_other", account_id=account_id
            )
            is None
        )
        assert (
            await queries.find_parked_servicer(
                conn, caller_session_id="ses_nobody", tool_call_id="tc_real", account_id=account_id
            )
            is None
        )


# ─── find_and_repair_ghosts routing ──────────────────────────────────────────


async def test_sweep_reparks_resumable_ghost_routing(
    env: tuple[asyncpg.Pool[Any], str, str], monkeypatch: Any
) -> None:
    """A ``call_*`` ghost with a live edge is RE-PARKED (not error-repaired); one with no
    edge → retryable ``launch_lost``; a ``bash`` ghost → the existing error-repair."""
    pool, account_id, _ = env
    caller = await _seed_session(
        pool, "route-caller", tools=[ToolSpec(type="call_session"), ToolSpec(type="bash")]
    )
    servicer = await _seed_session(pool, "route-servicer", tools=[ToolSpec(type="bash")])
    # tc_live: call_session with a durable edge → re-park. tc_lost: call_session, NO edge →
    # launch_lost. tc_bash: a side-effectful ghost → unchanged error-repair.
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=caller,
            kind="message",
            data=_assistant(
                [("tc_live", "call_session"), ("tc_lost", "call_session"), ("tc_bash", "bash")]
            ),
        )
    await _write_session_edge(
        pool,
        servicer_id=servicer,
        caller_session_id=caller,
        tool_call_id="tc_live",
        request_id="req_live",
    )

    relaunch = mock.Mock()
    monkeypatch.setattr("aios.harness.tool_dispatch.relaunch_parked_task", relaunch)

    repaired = await find_and_repair_ghosts(
        pool, runtime.require_inflight_tool_registry(), session_id=caller
    )

    # tc_live: re-parked — relaunch called with the session servicer handle, NOT error-repaired.
    relaunch.assert_called_once()
    kwargs = relaunch.call_args.kwargs
    assert kwargs["servicer_kind"] == "session"
    assert kwargs["servicer_id"] == servicer
    assert kwargs["request_id"] == "req_live"
    assert kwargs["call"]["id"] == "tc_live"
    repaired_ids = {tcid for _sid, tcid in repaired}
    assert "tc_live" not in repaired_ids

    # tc_lost (no edge) and tc_bash both get an error-repair result.
    assert {"tc_lost", "tc_bash"} <= repaired_ids
    lost = await _tool_results(pool, caller, "tc_lost")
    assert len(lost) == 1 and lost[0]["is_error"] is True
    assert "did not start" in lost[0]["content"]
    bash = await _tool_results(pool, caller, "tc_bash")
    assert len(bash) == 1 and bash[0]["is_error"] is True
    # tc_live got NO synthetic result (it is being re-parked, not error-repaired).
    assert await _tool_results(pool, caller, "tc_live") == []


# ─── the resume mechanism (span-less, lands the servicer answer) ──────────────


async def test_resume_parked_lands_answer_without_start_span(
    env: tuple[asyncpg.Pool[Any], str, str], monkeypatch: Any
) -> None:
    """``_resume_parked_async`` re-parks a call whose servicer already answered (the
    answered-during-downtime case) and lands the answer as the original tool result — and
    writes NO ``tool_execute_start`` span (the span-less bracket)."""
    from aios.harness import tool_dispatch

    pool, account_id, db_url = env
    caller = await _seed_session(pool, "resume-caller", tools=[ToolSpec(type="call_session")])
    servicer = await _seed_session(pool, "resume-servicer", tools=[ToolSpec(type="bash")])
    await _write_session_edge(
        pool,
        servicer_id=servicer,
        caller_session_id=caller,
        tool_call_id="tc_done",
        request_id="req_done",
    )
    # The servicer already answered during the worker's downtime.
    await workflow_completion.respond_to_request(
        pool, servicer, request_id="req_done", outcome=Ok(result={"v": 42})
    )
    # _park_on_task reads the LISTEN db_url off settings; point it at the test DB.
    monkeypatch.setattr(
        "aios.tools.invoke_session.get_settings", lambda: SimpleNamespace(db_url=db_url)
    )

    await tool_dispatch._resume_parked_async(
        pool,
        caller,
        {"id": "tc_done", "function": {"name": "call_session", "arguments": "{}"}},
        servicer_kind="session",
        servicer_id=servicer,
        request_id="req_done",
        output_schema=None,
        account_id=account_id,
    )

    results = await _tool_results(pool, caller, "tc_done")
    assert len(results) == 1
    assert results[0].get("is_error") is not True
    assert json.loads(results[0]["content"]) == {"ok": {"v": 42}}

    # Span-less: the re-park wrote no tool_execute_start span, so a later ghost
    # classification can't be misled into the "may have completed" branch (#1431 hazard).
    async with pool.acquire() as conn:
        spans = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'span' "
            "AND data->>'event' = 'tool_execute_start' AND data->>'tool_call_id' = $2",
            caller,
            "tc_done",
        )
    assert spans == 0


async def test_recovery_isolates_per_ghost_failure(
    env: tuple[asyncpg.Pool[Any], str, str], monkeypatch: Any
) -> None:
    """A transient failure resolving one resumable ghost must NOT abort recovery of the
    others — the cross-session sweep needs the same per-item isolation the error-repair
    loop has, or one gone caller strands every other tenant's parked task."""
    pool, account_id, _ = env
    caller = await _seed_session(pool, "iso-caller", tools=[ToolSpec(type="call_session")])
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=caller,
            kind="message",
            data=_assistant([("tc_boom", "call_session"), ("tc_ok", "call_session")]),
        )

    async def flaky(
        conn: Any, *, caller_session_id: str, tool_call_id: str, account_id: str
    ) -> Any:
        if tool_call_id == "tc_boom":
            raise RuntimeError("transient db error")
        return None  # tc_ok → no edge → launch_lost

    monkeypatch.setattr("aios.harness.sweep.find_parked_servicer", flaky)

    # Must return normally despite tc_boom raising mid-loop.
    repaired = await find_and_repair_ghosts(
        pool, runtime.require_inflight_tool_registry(), session_id=caller
    )
    repaired_ids = {tcid for _sid, tcid in repaired}

    # The healthy ghost was still recovered (launch_lost error), the failed one isolated
    # (no result — left for the next sweep), and recovery did not abort.
    assert "tc_ok" in repaired_ids
    assert "tc_boom" not in repaired_ids
    assert await _tool_results(pool, caller, "tc_boom") == []
    ok = await _tool_results(pool, caller, "tc_ok")
    assert len(ok) == 1 and ok[0]["is_error"] is True
