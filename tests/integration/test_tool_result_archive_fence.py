"""Integration regression for #1823 C2 spec test (l): **tool result after the
Phase B ``archived_at`` flip ⇒ clean drop** — real DB, real archive path, no
mocks of the queries under test.

The unit-tier ``TestArchiveFenceRace`` (tests/unit/test_tool_dispatch.py) pins
the fence's *branch logic* but stubs ``_append_tool_result`` / ``_trigger_sweep``
/ ``load_live_session_account_id``, so it cannot verify that an ACTUAL Phase B
archive flip makes the REAL lifecycle append raise the archived-fence
``NotFoundError`` and that the REAL liveness predicate then converts exactly
that fence into a clean drop. This test closes that fidelity gap:

* a real session (real account/agent/env rows) with a real awaited inbound
  request and a real cancel-marker;
* a real in-flight tool task — the production ``_execute_tool_async`` driving
  the actual ``_tool_lifecycle``, whose ``tool_execute_start`` span commits to
  the DB BEFORE the flip;
* the REAL two-phase teardown mid-flight: Phase A
  (``harvest_session_cancel_markers``, classifies teardown) then Phase B
  (``finalize_session_cancel_markers``, flips ``archived_at`` under the session
  row lock — the same transactional flip ``archive_session_conn`` performs);
* only then does the tool body finish, so the real result append
  (``_append_tool_result_event`` → ``queries.append_event``) lands AFTER the
  flip and hits the real ``archived_at IS NULL`` fence.

Assertions are real-DB: the lifecycle task exits cleanly (no ``NotFoundError``
escapes; on the cancel arm the ``CancelledError`` still propagates), and NO
post-flip event row — in particular no ``role:"tool"`` result row — exists for
the session.

The ONLY test double is the tool handler body itself (a fake
``invoke_builtin`` that blocks on an asyncio event so the start-flip-result
interleaving is deterministic) plus the procrastinate wake deferrals (no open
App in this tier — the same scaffolding ``test_session_cancel_leaf.py`` uses).
Neither is a query under test.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime, tool_dispatch
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.services import sessions as service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_tool_fence"
_TOOL_CALL_ID = "tc_fence"
_TOOL_NAME = "demo"


@pytest.fixture(autouse=True)
def _mock_prompt_wakes() -> Iterator[None]:
    """Phase A fires best-effort prompt procrastinate wakes post-commit (no open App in
    this tier) — patch the deferral bindings exactly as ``test_session_cancel_leaf.py``
    does. The durable seed + the fence queries under test are untouched."""
    with (
        patch("aios.services.sessions.defer_wake", new=AsyncMock()),
        patch("aios.services.sessions.defer_run_wake", new=AsyncMock()),
    ):
        yield


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'tool-fence')",
                _ACCOUNT,
            )
        _agent, env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="tool-fence"
        )
        yield pool, session.id, env.id
    finally:
        await pool.close()


async def _seed_teardown_ready_session(
    pool: asyncpg.Pool[Any], session_id: str, env_id: str
) -> None:
    """Make the session an owned teardown target with one revoked inbound: the
    ``archive_when_idle`` ownership flag, a sole awaited inbound request, and a
    cancel-marker on it — so Phase A classifies ``teardown=True`` and Phase B archives."""
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET archive_when_idle = TRUE WHERE id = $1", session_id)
        await queries.append_request_opened(
            conn,
            session_id=session_id,
            account_id=_ACCOUNT,
            request_id="req_teardown",
            caller={"kind": "api", "id": _ACCOUNT},
            depth=0,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            awaited=True,
        )
        await queries.insert_session_cancel_marker(
            conn, session_id=session_id, request_id="req_teardown", account_id=_ACCOUNT
        )
        # The assistant turn that issued the in-flight call — the real event the
        # result append's parent lookup and open_tool_call_count accounting key on.
        await queries.append_event(
            conn,
            account_id=_ACCOUNT,
            session_id=session_id,
            kind="message",
            data={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": _TOOL_CALL_ID,
                        "type": "function",
                        "function": {"name": _TOOL_NAME, "arguments": "{}"},
                    }
                ],
            },
        )


@pytest.mark.parametrize("outcome", ["success", "handler_error", "cancelled"])
async def test_tool_result_after_phase_b_flip_is_clean_drop(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    monkeypatch: Any,
    outcome: str,
) -> None:
    """Spec §6 test (l): a tool result landing after the Phase B flip is a clean drop.

    Three real post-flip result appends, one per lifecycle arm: the success-path
    body append (``outcome="success"``), the generic-exception arm's error append
    (``outcome="handler_error"``), and the in-body cancel arm's ``cancelled`` append
    (``outcome="cancelled"`` — where the ``CancelledError`` must still propagate
    while its append's archived fence is dropped)."""
    pool, session_id, env_id = pool_and_session
    await _seed_teardown_ready_session(pool, session_id, env_id)
    monkeypatch.setattr(runtime, "inflight_tool_registry", InflightToolRegistry())

    tool_started = asyncio.Event()
    release = asyncio.Event()

    async def fake_invoke_builtin(
        _session_id: str, _name: str, _raw_args: Any, *, tool_call_id: str
    ) -> dict[str, Any]:
        # Test-harness tool body: block so the Phase B flip deterministically lands
        # between the (real) tool_execute_start span and the (real) result append.
        tool_started.set()
        await release.wait()
        if outcome == "handler_error":
            raise RuntimeError("handler failed after the flip")
        return {"ok": True}

    monkeypatch.setattr(tool_dispatch, "invoke_builtin", fake_invoke_builtin)

    call = {
        "id": _TOOL_CALL_ID,
        "type": "function",
        "function": {"name": _TOOL_NAME, "arguments": "{}"},
    }
    task = asyncio.create_task(
        tool_dispatch._execute_tool_async(
            pool, session_id, call, account_id=_ACCOUNT, parent_focal_at_arrival=None
        )
    )
    await asyncio.wait_for(tool_started.wait(), timeout=15)

    # The real start span committed BEFORE the flip — the (l) premise: tool start
    # pre-flip, tool result post-flip.
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM events WHERE session_id=$1 AND kind='span' "
                "AND data->>'event'='tool_execute_start' AND data->>'tool_call_id'=$2",
                session_id,
                _TOOL_CALL_ID,
            )
            == 1
        )

    # THE FLIP, mid-flight, through the real two-phase teardown: Phase A classifies,
    # Phase B archives under the session row lock (real ``archived_at`` UPDATE).
    decision = await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)
    assert decision is not None and decision.teardown
    assert await service.finalize_session_cancel_markers(
        pool, session_id, account_id=_ACCOUNT, teardown=True, request_ids=decision.request_ids
    )
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT archived_at FROM sessions WHERE id=$1", session_id)
            is not None
        )
        events_at_flip = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id=$1", session_id
        )

    # Release the tool: its REAL result append (and span-end/sweep tails) now land
    # after the flip and hit queries.append_event's archived fence.
    if outcome == "cancelled":
        # The cancel strikes the blocked body; the cancel arm's fenced ``cancelled``
        # append hits the archived fence, is dropped, and the CancelledError STILL
        # propagates — never replaced by an escaping NotFoundError.
        assert task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        release.set()
        # Clean exit: no NotFoundError (nor the handler's RuntimeError) escapes —
        # this await raising anything fails the test.
        await asyncio.wait_for(task, timeout=15)

    async with pool.acquire() as conn:
        # THE (l) POSTCONDITION, on the real ledger: no post-flip tool-result row…
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM events WHERE session_id=$1 AND kind='message' "
                "AND role='tool'",
                session_id,
            )
            == 0
        )
        # …and nothing else landed post-flip either (the span-end + sweep tails were
        # fenced too): the archived session's log is frozen at the flip.
        assert (
            await conn.fetchval("SELECT count(*) FROM events WHERE session_id=$1", session_id)
            == events_at_flip
        )
        # The session is still archived (the drop never resurrected anything).
        assert (
            await conn.fetchval("SELECT archived_at FROM sessions WHERE id=$1", session_id)
            is not None
        )
