"""Integration tests for ``defer_obligations`` — the ACTIVE-WAIT semantics (#1533).

DB-backed (testcontainer Postgres). The tool is deliberately NOT a new
mechanism: it is an ordinary in-flight tool call that takes ``duration_seconds``
to return, parked on the session's own event channel (the ``_await_session``
triad). These tests pin the guarantees the design says are INHERITED from how
in-flight tool calls already work:

* **active + un-nudged** — a turn calling ``defer_obligations`` while owing an
  awaited obligation does NOT trip the quiescence guard (no nudge, no
  auto-``no_return``); the session derives ``active`` and the obligation stays
  open, untouched (session-wide suppression by inaction — no guard change);
* **duration elapse** — the real dispatch path lands exactly one tool-role
  result ``resolved="duration_elapsed"`` after the timeout and
  ``open_tool_call_count`` returns to baseline;
* **early stimulus** — an inbound message mid-wait resolves the park early with
  ``resolved="stimulus"`` and exactly ONE tool-role event for the call id (no
  parallel re-launch);
* **no redispatch** — a defer call never appears in
  ``list_confirmed_unresolved_tool_calls`` (no ``tool_confirmed`` event is ever
  written for it), so the confirmed-redispatch path can't re-run the handler;
* **crash → launch_lost** — a ghosted (worker-crashed) in-flight defer takes the
  resumable branch, finds no servicer edge (``find_parked_servicer`` → ``None``),
  and lands the retryable ``launch_lost`` result — never a re-park and never the
  pessimistic may-have-completed branch.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.sweep import find_and_repair_ghosts
from aios.harness.tool_dispatch import launch_tool_calls
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from aios.services.await_completion import await_completion as real_await_completion
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_defer"
_FROZEN: dict[str, list[Any]] = {"tools": [], "mcp_servers": [], "http_servers": []}


@pytest.fixture
async def env(
    migrated_db_url: str, _reset_db_state: None, monkeypatch: Any
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, account_id)`` with the worker runtime wired so the real
    dispatch path (``launch_tool_calls`` → ``_tool_lifecycle`` → handler) and
    the ghost sweep can run without a live worker (wakes patched out).

    The handler resolves its LISTEN url from settings; point it at the test DB
    (same stance as ``test_durable_await_resume``'s ``_park_on_task`` patch).
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev_pool, prev_reg = runtime.pool, runtime.inflight_tool_registry
    runtime.pool = pool
    runtime.inflight_tool_registry = InflightToolRegistry()
    monkeypatch.setattr(
        "aios.tools.defer_obligations.get_settings",
        lambda: SimpleNamespace(db_url=migrated_db_url),
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'defer-root')",
                _ACCOUNT,
            )
        with (
            mock.patch("aios.services.sessions.defer_wake", new=AsyncMock()),
            mock.patch("aios.jobs.app.defer_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
        ):
            yield pool, _ACCOUNT
    finally:
        runtime.pool, runtime.inflight_tool_registry = prev_pool, prev_reg
        await pool.close()


def _defer_call(call_id: str, duration_seconds: int) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "defer_obligations",
            "arguments": json.dumps({"duration_seconds": duration_seconds}),
        },
    }


async def _seed_session(pool: asyncpg.Pool[Any], prefix: str) -> str:
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=_ACCOUNT, prefix=prefix, tools=[ToolSpec(type="defer_obligations")]
    )
    return session.id


async def _append_defer_turn(
    pool: asyncpg.Pool[Any], session_id: str, call: dict[str, Any]
) -> None:
    """Append an assistant turn carrying ``call`` through the REAL quiescence
    guard — the exact end-of-turn write path the harness loop uses."""
    session = await sessions_service.get_session_basic(pool, session_id, account_id=_ACCOUNT)
    assistant = {
        "role": "assistant",
        "content": "",
        "reacting_to": session.last_event_seq,
        "tool_calls": [call],
    }
    result = await sessions_service.append_assistant_and_guard_quiescence(
        pool, session_id, assistant, account_id=_ACCOUNT
    )
    assert not result.nudged and result.autoerror_caller_run_id is None


async def _open_obligation(pool: asyncpg.Pool[Any], session_id: str, request_id: str) -> None:
    """Write an awaited ``request_opened`` edge so ``session_id`` owes a response."""
    caller = await _seed_session(pool, f"caller-{request_id}")
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            caller={"kind": "session", "id": caller},
            depth=1,
            environment_id="env_x",
            frozen_surface=_FROZEN,
            vault_ids=[],
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


async def _wait_for_result(
    pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str, *, deadline_seconds: float
) -> list[dict[str, Any]]:
    deadline = asyncio.get_running_loop().time() + deadline_seconds
    while asyncio.get_running_loop().time() < deadline:
        results = await _tool_results(pool, session_id, tool_call_id)
        if results:
            return results
        await asyncio.sleep(0.1)
    return await _tool_results(pool, session_id, tool_call_id)


# ─── active + un-nudged while owing (the inherited guard behavior) ────────────


async def test_defer_turn_owing_obligation_is_unnudged_and_active(
    env: tuple[asyncpg.Pool[Any], str],
) -> None:
    """A turn that calls ``defer_obligations`` while owing an awaited obligation
    never reaches the nudge/``no_return`` loop: the in-flight tool call keeps the
    session ACTIVE, the obligation stays open UNTOUCHED (session-wide suppression
    by inaction — no persisted snooze, no guard change), zero nudges."""
    pool, account_id = env
    sid = await _seed_session(pool, "unnudged")
    await _open_obligation(pool, sid, "req_defer")

    await _append_defer_turn(pool, sid, _defer_call("tc_guard", 60))

    async with pool.acquire() as conn:
        open_ids = await queries.get_open_request_ids(conn, sid, account_id=account_id)
        status = await queries.derive_session_status(conn, sid, account_id=account_id)
        nudges = await queries.count_request_nudges(
            conn, sid, account_id=account_id, request_id="req_defer"
        )
        nudge_msgs = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'message' "
            "AND role = 'user' AND data->'metadata'->'nudged_request_ids' IS NOT NULL",
            sid,
        )
        no_returns = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'lifecycle' "
            "AND data->>'event' = 'request_response'",
            sid,
        )
    assert open_ids == ["req_defer"]  # still open — deferring is not answering
    assert status == "active"  # open in-flight tool call → never idle
    assert nudges == 0 and nudge_msgs == 0  # actively un-nudged
    assert no_returns == 0  # no auto-error


# ─── duration elapse (the real dispatch path) ─────────────────────────────────


async def test_duration_elapse_returns_single_result(env: tuple[asyncpg.Pool[Any], str]) -> None:
    pool, account_id = env
    sid = await _seed_session(pool, "elapse")
    call = _defer_call("tc_elapse", 1)
    await _append_defer_turn(pool, sid, call)

    launch_tool_calls(pool, sid, [call], account_id=account_id)
    results = await _wait_for_result(pool, sid, "tc_elapse", deadline_seconds=20)

    assert len(results) == 1
    assert results[0].get("is_error") is not True
    assert json.loads(results[0]["content"]) == {"deferred": True, "resolved": "duration_elapsed"}
    async with pool.acquire() as conn:
        open_count = await conn.fetchval(
            "SELECT open_tool_call_count FROM sessions WHERE id = $1", sid
        )
    assert open_count == 0  # back to baseline — the turn is closed


# ─── early resolve on an inbound stimulus ─────────────────────────────────────


async def test_inbound_stimulus_resolves_early(
    env: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
) -> None:
    """A message landing mid-wait ends the defer immediately (``resolved=
    "stimulus"``) — long before the 600s duration — with exactly ONE tool-role
    event for the call id (no parallel re-launch)."""
    pool, account_id = env
    sid = await _seed_session(pool, "stim")

    # Signal the moment the handler parks (subscription open + baseline read),
    # so the inbound message deterministically lands DURING the wait.
    parked = asyncio.Event()

    async def _signalling_await_completion(queue: Any, **kwargs: Any) -> Any:
        parked.set()
        return await real_await_completion(queue, **kwargs)

    monkeypatch.setattr(
        "aios.tools.defer_obligations.await_completion", _signalling_await_completion
    )

    call = _defer_call("tc_stim", 600)
    await _append_defer_turn(pool, sid, call)
    launch_tool_calls(pool, sid, [call], account_id=account_id)
    await asyncio.wait_for(parked.wait(), timeout=10)

    # While parked the session is ACTIVE (this is what suppresses nudging).
    async with pool.acquire() as conn:
        assert await queries.derive_session_status(conn, sid, account_id=account_id) == "active"
        # The inbound stimulus: bumps last_stimulus_seq and NOTIFYs the channel.
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=sid,
            kind="message",
            data={"role": "user", "content": "wake up — news arrived"},
        )

    results = await _wait_for_result(pool, sid, "tc_stim", deadline_seconds=15)
    assert len(results) == 1  # exactly one tool-role event — no parallel re-launch
    assert results[0].get("is_error") is not True
    assert json.loads(results[0]["content"]) == {"deferred": True, "resolved": "stimulus"}


# ─── no redispatch ────────────────────────────────────────────────────────────


async def test_defer_call_is_not_redispatchable(env: tuple[asyncpg.Pool[Any], str]) -> None:
    """No ``tool_confirmed`` event is ever written for a defer call, so the
    confirmed-redispatch resolver never offers it — crash recovery is exclusively
    the resumable ghost sweep (which re-parks nothing and lands ``launch_lost``),
    never a handler re-run."""
    pool, account_id = env
    sid = await _seed_session(pool, "nore")
    await _append_defer_turn(pool, sid, _defer_call("tc_nore", 60))

    dispatchable = await sessions_service.list_confirmed_unresolved_tool_calls(
        pool, sid, account_id=account_id
    )
    assert dispatchable == []


# ─── crash recovery: resumable ghost → launch_lost ────────────────────────────


async def test_crashed_defer_ghost_lands_launch_lost(
    env: tuple[asyncpg.Pool[Any], str], monkeypatch: Any
) -> None:
    """A worker crash mid-wait leaves an in-flight defer with no result and no
    task. The sweep's RESUMABLE branch (``resumable=True``) consults
    ``find_parked_servicer``, finds no edge (a defer writes none), and lands the
    retryable ``launch_lost`` result — no re-park, no may-have-completed. The
    wait truncates cleanly; the model can simply re-defer."""
    pool, _account_id = env
    sid = await _seed_session(pool, "ghost")
    # The assistant turn exists but its task is gone — the crash shape.
    await _append_defer_turn(pool, sid, _defer_call("tc_ghost", 600))

    relaunch = mock.Mock()
    monkeypatch.setattr("aios.harness.tool_dispatch.relaunch_parked_task", relaunch)

    repaired = await find_and_repair_ghosts(
        pool, runtime.require_inflight_tool_registry(), session_id=sid
    )

    assert "tc_ghost" in {tcid for _sid, tcid in repaired}
    relaunch.assert_not_called()  # nothing to re-park — a defer has no servicer edge
    results = await _tool_results(pool, sid, "tc_ghost")
    assert len(results) == 1 and results[0]["is_error"] is True
    # Pin the BRANCH, not just is_error: the launch_lost wording (retryable,
    # "nothing was launched"), not the pessimistic "may have completed" one.
    assert "did not start before the worker restarted" in results[0]["content"]
    assert "may have completed" not in results[0]["content"]
