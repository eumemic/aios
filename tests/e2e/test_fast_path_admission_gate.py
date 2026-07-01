"""Equivalence + safety tests for the per-turn fast-path admission gate (#1659).

The per-turn entry guard (``harness/loop.py``) now runs a cheap PK-scoped
``session_has_pending_work`` gate BEFORE the multi-CTE
``find_sessions_needing_inference``. To be safe against the wedge class
(a missed wake → a stuck session), the gate must be a proven
**over-approximation** of the full sweep:

    fast_path_says_no_work  ⟹  full_sweep_says_no_work

i.e. the gate may only early-out ("no work") when there is *provably* no
pending work; on any "maybe work" it falls through to the full sweep. A wrong
predicate is then bounded to an occasional *extra* full sweep (cheap), NEVER a
missed wake (unbounded — a wedged session).

These tests lock that implication across a matrix of session states:

- pending unreacted user stimulus            (active → both say work)
- fully reacted / idle                        (both say no work)
- open (unresolved) tool call                 (active via open_tool_call_count)
- confirmed-but-unresolved tool call          (case (c) ⊆ active)
- errored + unreacted                         (parked → both say no work)
- archived                                    (fenced out → both say no work)
- unharvested cancel-marker on an idle session (the ONE full-sweep path the
  active predicate does not subsume — the gate must still say "maybe work")

The equivalence assertion is the core fence; the directional "gate found it
too" assertions document why the early-out is safe on each shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.sweep import (
    find_sessions_needing_inference,
    session_has_pending_work,
)
from aios.models.events import ERRORED_LIFECYCLE_STATUS, ERRORED_LIFECYCLE_STOP_REASON
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker

pytestmark = pytest.mark.docker

_ACCOUNT = "acc_test_stub"


async def _ensure_agent_and_env(pool: asyncpg.Pool[Any]) -> tuple[str, str]:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agents (id, name, model, account_id) "
            "VALUES ('agt_fp', 'fp', 'openrouter/x', 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING"
        )
        await conn.execute(
            "INSERT INTO environments (id, name, account_id) "
            "VALUES ('env_fp', 'env_fp', 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING"
        )
    return "agt_fp", "env_fp"


async def _new_session(pool: asyncpg.Pool[Any]) -> str:
    agent_id, environment_id = await _ensure_agent_and_env(pool)
    session = await sessions_service.create_session(
        pool,
        agent_id=agent_id,
        environment_id=environment_id,
        title=None,
        metadata={},
        account_id=_ACCOUNT,
    )
    return session.id


async def _append_assistant_reacting_to_tail(pool: asyncpg.Pool[Any], sid: str) -> None:
    """Append an assistant message reacting to the current tail, driving the
    session idle (``last_reacted_seq`` catches up to ``last_stimulus_seq``)."""
    async with pool.acquire() as conn:
        tail = await conn.fetchval(
            "SELECT last_event_seq FROM sessions WHERE id = $1", sid
        )
    await sessions_service.append_event(
        pool,
        sid,
        "message",
        {"role": "assistant", "content": "ok", "reacting_to": tail},
        account_id=_ACCOUNT,
    )


async def _fast_path(pool: asyncpg.Pool[Any], sid: str) -> bool:
    async with pool.acquire() as conn:
        return await session_has_pending_work(conn, sid)


async def _full_sweep_has_work(pool: asyncpg.Pool[Any], sid: str) -> bool:
    result = await find_sessions_needing_inference(
        pool, InflightToolRegistry(), session_id=sid
    )
    return sid in result


@pytest.fixture
async def db_pool(aios_env: dict[str, str]) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(aios_env["AIOS_DB_URL"], min_size=1, max_size=2)
    try:
        yield pool
    finally:
        await pool.close()


# ─── state builders (one per matrix row) ─────────────────────────────────────


async def _state_pending_user(pool: asyncpg.Pool[Any]) -> str:
    sid = await _new_session(pool)
    await sessions_service.append_user_message(pool, sid, "hi", account_id=_ACCOUNT)
    return sid


async def _state_idle(pool: asyncpg.Pool[Any]) -> str:
    sid = await _new_session(pool)
    await sessions_service.append_user_message(pool, sid, "hi", account_id=_ACCOUNT)
    await _append_assistant_reacting_to_tail(pool, sid)
    return sid


async def _state_open_tool_call(pool: asyncpg.Pool[Any]) -> str:
    sid = await _new_session(pool)
    await sessions_service.append_user_message(pool, sid, "run it", account_id=_ACCOUNT)
    # Assistant message with a tool_call that has no result → open_tool_call_count = 1.
    await sessions_service.append_event(
        pool,
        sid,
        "message",
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_open",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
            "reacting_to": 1,
        },
        account_id=_ACCOUNT,
    )
    return sid


async def _state_confirmed_unresolved(pool: asyncpg.Pool[Any]) -> str:
    sid = await _state_open_tool_call(pool)
    # A ``tool_confirmed``/``allow`` lifecycle for the still-open call — case (c).
    await sessions_service.append_event(
        pool,
        sid,
        "lifecycle",
        {"event": "tool_confirmed", "result": "allow", "tool_call_id": "tc_open"},
        account_id=_ACCOUNT,
    )
    return sid


async def _state_errored(pool: asyncpg.Pool[Any]) -> str:
    sid = await _new_session(pool)
    await sessions_service.append_user_message(pool, sid, "hi", account_id=_ACCOUNT)
    await sessions_service.append_event(
        pool,
        sid,
        "lifecycle",
        {
            "event": "turn_ended",
            "status": ERRORED_LIFECYCLE_STATUS,
            "stop_reason": ERRORED_LIFECYCLE_STOP_REASON,
        },
        account_id=_ACCOUNT,
    )
    return sid


async def _state_archived(pool: asyncpg.Pool[Any]) -> str:
    sid = await _state_pending_user(pool)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET archived_at = now() WHERE id = $1", sid
        )
    return sid


async def _state_idle_with_cancel_marker(pool: asyncpg.Pool[Any]) -> str:
    """An IDLE session (no active stimulus) carrying an unharvested cancel-marker.

    This is the ONE full-sweep return path ``session_active_predicate`` does not
    subsume (it is UNIONed BELOW the errored subtraction), so it is the load-
    bearing case for the OR-ed marker EXISTS in the fast-path gate: an
    over-approximation that dropped it would let the gate early-out on a session
    the full sweep would return — a missed cancel-leaf.
    """
    sid = await _state_idle(pool)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO session_cancel_markers (session_id, request_id, account_id) "
            "VALUES ($1, 'req_cancel', $2)",
            sid,
            _ACCOUNT,
        )
    return sid


_STATE_BUILDERS = {
    "pending_user": _state_pending_user,
    "idle": _state_idle,
    "open_tool_call": _state_open_tool_call,
    "confirmed_unresolved": _state_confirmed_unresolved,
    "errored": _state_errored,
    "archived": _state_archived,
    "idle_with_cancel_marker": _state_idle_with_cancel_marker,
}

# States where BOTH the fast path and the full sweep must agree there is work.
_HAS_WORK_STATES = {
    "pending_user",
    "open_tool_call",
    "confirmed_unresolved",
    "idle_with_cancel_marker",
}
# States where the full sweep returns no work.
_NO_WORK_STATES = {"idle", "errored", "archived"}


@needs_docker
class TestFastPathOverApproximation:
    @pytest.mark.parametrize("state", sorted(_STATE_BUILDERS))
    async def test_gate_is_over_approximation_of_full_sweep(
        self, db_pool: asyncpg.Pool[Any], state: str
    ) -> None:
        """Core safety fence: ``fast_path_says_no_work ⟹ full_sweep_says_no_work``.

        Whenever the cheap gate early-outs ("no work"), the authoritative full
        sweep must also find no work — so the early-out can never drop a wake.
        """
        sid = await _STATE_BUILDERS[state](db_pool)

        gate = await _fast_path(db_pool, sid)
        full = await _full_sweep_has_work(db_pool, sid)

        if not gate:
            assert not full, (
                f"OVER-APPROXIMATION VIOLATED for state={state!r}: the fast-path gate "
                f"said 'no work' but the full sweep would wake the session — this is a "
                f"MISSED WAKE (wedge-class bug). The gate must be TRUE whenever the "
                f"full sweep could return the session."
            )

    @pytest.mark.parametrize("state", sorted(_HAS_WORK_STATES))
    async def test_has_work_states_fall_through(
        self, db_pool: asyncpg.Pool[Any], state: str
    ) -> None:
        """On every state the full sweep would wake, the gate says "maybe work"
        (True) → the guard falls through to the full sweep. This exercises the
        fall-through path (the uncertain branch) for each work-bearing shape."""
        sid = await _STATE_BUILDERS[state](db_pool)

        assert await _full_sweep_has_work(db_pool, sid), (
            f"fixture bug: state={state!r} was expected to be work-bearing"
        )
        assert await _fast_path(db_pool, sid), (
            f"state={state!r} needs work but the gate early-outs — the guard would "
            f"skip the full sweep and MISS the wake."
        )

    @pytest.mark.parametrize("state", sorted(_NO_WORK_STATES))
    async def test_no_work_states_early_out(
        self, db_pool: asyncpg.Pool[Any], state: str
    ) -> None:
        """On genuinely-quiescent states the gate early-outs (False) — the whole
        point of the fast path (skip the multi-CTE sweep on wasted wakes)."""
        sid = await _STATE_BUILDERS[state](db_pool)

        assert not await _full_sweep_has_work(db_pool, sid), (
            f"fixture bug: state={state!r} was expected to be quiescent"
        )
        assert not await _fast_path(db_pool, sid), (
            f"state={state!r} has no work but the gate did not early-out — the fast "
            f"path buys nothing here (wasted-wake path stays on the heavy sweep)."
        )

    async def test_missing_session_is_no_work(self, db_pool: asyncpg.Pool[Any]) -> None:
        """A gone/unknown session id → no row → no work (idempotent no-op wake)."""
        assert not await _fast_path(db_pool, "sess_does_not_exist")
