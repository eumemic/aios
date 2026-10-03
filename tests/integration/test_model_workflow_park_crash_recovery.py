"""Crash-recovery for the model-dispatch park (#1635).

The session-side park introduced by the ``workflow:`` model binding (#1634) needs
its own crash-recovery. The existing recovery is **tool-call-bound**:
``find_parked_servicer`` keys on a ``tool_call_id`` and the ghost-scan re-parks only
servicers backed by an open tool_call in a *persisted assistant message*. A
model-dispatch park has **neither** — the assistant message is the bound workflow's
output, produced only after the park resolves — so a worker crash while parked
**strands** the session: the inner run completes, but the in-process harvest task
died with the worker, so nothing writes the harvest event back and the turn never
completes.

These tests drive the REAL park → crash → sweep-repark → harvest path against a
testcontainer Postgres:

* ``test_crash_while_parked_is_recovered_by_sweep`` — park, simulate a crash (the
  harvest task never ran), the inner run resolves, then the crash-recovery sweep
  re-parks → the harvest lands and the turn completes with **no loss and no
  double-charge**.
* ``test_repark_is_idempotent_no_double_harvest`` — re-parking a park that has
  already been harvested writes NO second harvest (the dedup guard on ``run_id``),
  and folding still produces exactly one assistant turn.
* ``test_steady_state_park_is_not_double_parked`` — a park whose live harvest task is
  in-flight in THIS worker is NOT re-parked by the sweep (the in-flight key gate).
* ``test_consumed_park_is_not_reparked`` — a fully-folded (consumed) park is not a
  crash-recovery candidate.

As in ``test_model_workflow_park_sweep.py`` the fire-and-forget harvest poller is
patched to a no-op; the harvest is driven manually by resolving the inner run +
calling ``write_harvest_event`` — exactly the path the (lost) park task would take,
minus the LISTEN/NOTIFY plumbing. The crash-recovery re-park is exercised directly
via ``repark_stranded_model_dispatch``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import model_workflow as mwf
from aios.harness import runtime
from aios.harness.completion import LlmRequest
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.loop import run_session_step
from aios.harness.model_binding import WorkflowModelRef, parse_workflow_model
from aios.harness.model_workflow import write_harvest_event
from aios.harness.request_capture import forget_stored_blobs
from aios.harness.sweep import repark_stranded_model_dispatch
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from aios.services import workflows as wf_service
from aios.services.requests import Rebuilt, rebuild_request
from aios.workflows import run_tools
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration


class _EmptyToolProvider:
    async def list_tools_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> list[dict[str, Any]]:
        return []

    async def list_capabilities_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> dict[str, Any]:
        return {}


_INNER_COST_MICROUSD = 4321
_INNER_SCRIPT = (
    "async def main(input):\n"
    "    return {'content': 'recovered answer', 'tool_calls': [], 'finish_reason': '{finish_reason}'}\n"
)

_ACCOUNT = "acc_mwfcr"
_ENV = "env_mwfcr"


@pytest.fixture
async def mwf_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A pool on ``runtime.pool`` + an inflight registry + a seeded root tenant.

    ``defer_*wake`` enqueues are patched out (the steps are driven directly), and
    ``_launch_harvest_task`` is a no-op so the park does not spawn the background
    poller — simulating the worker crash (the in-process harvest task never runs).
    The crash-recovery re-park is driven explicitly in-test.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev_pool = runtime.pool
    prev_reg = runtime.inflight_tool_registry
    prev_tp = runtime.tool_provider
    runtime.pool = pool
    runtime.inflight_tool_registry = InflightToolRegistry()
    runtime.tool_provider = _EmptyToolProvider()
    mwf.reset_inflight_harvests()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'mwfcr-root')",
                _ACCOUNT,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'mwfcr-env', '{}'::jsonb, $2)",
                _ENV,
                _ACCOUNT,
            )
        run_tools._INFLIGHT.clear()
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
            mock.patch("aios.workflows.run_tools.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.sessions.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.harness.loop.defer_wake", new=AsyncMock()),
            mock.patch("aios.harness.loop.defer_run_wake", new=AsyncMock()),
            # Simulate the worker crash: the park's fire-and-forget harvest poller never
            # runs, so no harvest is ever written by the park task itself. The harvest is
            # driven manually below, and crash-recovery is exercised via the sweep.
            mock.patch("aios.harness.model_workflow._launch_harvest_task", new=mock.Mock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        mwf.reset_inflight_harvests()
        runtime.pool = prev_pool
        runtime.inflight_tool_registry = prev_reg
        runtime.tool_provider = prev_tp
        await pool.close()


async def _make_bound_session(
    pool: asyncpg.Pool[Any], *, finish_reason: str = "stop", output_model: str | None = None
) -> str:
    script = _INNER_SCRIPT.replace("{finish_reason}", finish_reason)
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id=_ACCOUNT,
            name="inner-model",
            script=script,
            output_model=output_model,
        )
    agent = await agents_service.create_agent(
        pool,
        account_id=_ACCOUNT,
        name="mwfcr-agent",
        model=f"workflow:{wf.id}",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    env = await environments_service.get_environment(pool, _ENV, account_id=_ACCOUNT)
    session = await sessions_service.create_session(
        pool,
        agent_id=agent.id,
        environment_id=env.id,
        title="mwfcr",
        metadata={},
        account_id=_ACCOUNT,
    )
    await sessions_service.append_user_message(pool, session.id, "answer this", account_id=_ACCOUNT)
    return session.id


async def _inner_run_ids(pool: asyncpg.Pool[Any], session_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id FROM wf_runs WHERE launcher_session_id = $1 AND account_id = $2 "
            "ORDER BY created_at",
            session_id,
            _ACCOUNT,
        )
    return [r["id"] for r in rows]


async def _run_caller(pool: asyncpg.Pool[Any], run_id: str) -> dict[str, Any]:
    async with pool.acquire() as conn:
        caller = await conn.fetchval(
            "SELECT caller FROM wf_runs WHERE id = $1 AND account_id = $2",
            run_id,
            _ACCOUNT,
        )
    return cast("dict[str, Any]", caller)


async def _account_spent(pool: asyncpg.Pool[Any]) -> int:
    async with pool.acquire() as conn:
        return await db_queries.get_account_spent_microusd(conn, _ACCOUNT)


async def _harvest_events(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(  # type: ignore[no-any-return]
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'span' "
            "AND data->>'event' = 'model_workflow_harvest'",
            session_id,
        )


async def _harvest_end_events(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(  # type: ignore[no-any-return]
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'span' "
            "AND data->>'event' = 'model_workflow_harvest_end'",
            session_id,
        )


async def _assistant_messages(pool: asyncpg.Pool[Any], session_id: str) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'message' "
            "AND role = 'assistant' ORDER BY seq",
            session_id,
        )
    return [r["data"] for r in rows]


async def _total_inner_call_llm_cost(pool: asyncpg.Pool[Any], session_id: str) -> int:
    total = 0
    async with pool.acquire() as conn:
        for run_id in await _inner_run_ids(pool, session_id):
            total += await wf_queries.get_run_call_llm_cost_microusd(
                conn, run_id, account_id=_ACCOUNT
            )
    return total


async def _resolve_inner_run(pool: asyncpg.Pool[Any], run_id: str) -> Any:
    """Run the inner workflow to completion + charge its ``call_llm`` meter once.

    This is the inner run reaching its terminal state — which is exactly what
    happens during the worker crash: the run finishes, but nobody on the SESSION
    side wrote the harvest back (the park task died). Returns the run's output.
    """
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        assert run.status == "completed", f"inner run not completed: {run.status}"
        await wf_queries.add_run_call_llm_cost_microusd(
            conn, run_id, _INNER_COST_MICROUSD, account_id=_ACCOUNT
        )
    return run.output


async def test_crash_while_parked_is_recovered_by_sweep(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """Kill the worker while parked → on restart the sweep re-parks, harvests, completes."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)

    # ── PARK (step N): one awaited inner run; the harvest task is a no-op (crash). ──
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    # The park's caller edge carries the tool_call_id-less model-dispatch discriminant —
    # the durable marker crash-recovery keys on (no assistant message, no tool_call_id).
    caller = await _run_caller(pool, inner_run_id)
    assert caller["kind"] == "session" and caller["id"] == session_id
    assert caller["purpose"] == "model_dispatch"

    # The worker crashed: NO harvest was written (the in-process task never ran), and
    # the session is stranded — the existing ghost scan can't see this park.
    assert await _harvest_events(pool, session_id) == 0
    async with pool.acquire() as conn:
        stranded = await db_queries.find_unharvested_model_dispatch_parks(conn)
    assert (session_id, inner_run_id, _ACCOUNT) in stranded, (
        "a crashed-while-parked session must be a crash-recovery candidate"
    )

    # The inner run resolves (it completed before / after the crash — its terminal state
    # is durable regardless). Nothing has written that state back to the session yet.
    run_output = await _resolve_inner_run(pool, inner_run_id)

    spent_after_run = await _account_spent(pool)

    # ── RESTART: the crash-recovery sweep re-derives the stranded park and re-parks. ──
    # Patch the (still-no-op) harvest task to a real harvest write — this stands in for
    # the re-launched ``_park_and_signal`` reading the run's terminal state and writing
    # the harvest, minus the LISTEN/NOTIFY poll.
    async def _fake_relaunched_harvest() -> None:
        await write_harvest_event(
            pool,
            session_id,
            run_id=inner_run_id,
            outcome="ok",
            output=run_output,
            error=None,
            account_id=_ACCOUNT,
        )

    with mock.patch.object(
        mwf, "_launch_harvest_task", side_effect=lambda *a, **k: None
    ) as launch_spy:
        reparked = await repark_stranded_model_dispatch(pool)
    assert reparked == 1, "the sweep must re-park exactly the one stranded session"
    launch_spy.assert_called_once()
    # The re-park launched a harvest task for the right (session, run) — drive it.
    _, kwargs = launch_spy.call_args
    assert kwargs["run_id"] == inner_run_id
    await _fake_relaunched_harvest()

    # The harvest event now exists; the next session step folds it into the turn.
    assert await _harvest_events(pool, session_id) == 1
    await run_session_step(session_id, cause="model_workflow_harvest")

    # The turn completed: exactly one assistant message carrying the inner answer.
    assistants = await _assistant_messages(pool, session_id)
    assert len(assistants) == 1
    assert assistants[0]["content"] == "recovered answer"
    assert await _harvest_end_events(pool, session_id) == 1, "the fold writes one consumed marker"

    # No double-charge: the harvest does not re-charge the outer turn, and the only
    # inference spend is the inner run's, charged ONCE (no second run was launched).
    assert await _account_spent(pool) == spent_after_run, "harvest must not re-charge"
    assert await _total_inner_call_llm_cost(pool, session_id) == _INNER_COST_MICROUSD
    assert await _inner_run_ids(pool, session_id) == [inner_run_id], (
        "crash-recovery must NOT launch a second inner run"
    )

    # The session is fully recovered — no longer a crash-recovery candidate.
    async with pool.acquire() as conn:
        stranded_after = await db_queries.find_unharvested_model_dispatch_parks(conn)
    assert session_id not in [s[0] for s in stranded_after]


async def test_repark_is_idempotent_no_double_harvest(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """Re-parking a park whose harvest already landed writes NO second harvest."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    run_output = await _resolve_inner_run(pool, inner_run_id)
    # The harvest already landed (e.g. a racing task won, or a prior re-park).
    await write_harvest_event(
        pool,
        session_id,
        run_id=inner_run_id,
        outcome="ok",
        output=run_output,
        error=None,
        account_id=_ACCOUNT,
    )
    assert await _harvest_events(pool, session_id) == 1

    # A park WITH a harvest is no longer stranded — the sweep does not re-park it.
    async with pool.acquire() as conn:
        stranded = await db_queries.find_unharvested_model_dispatch_parks(conn)
    assert session_id not in [s[0] for s in stranded], (
        "a harvested park is not a crash-recovery candidate"
    )
    reparked = await repark_stranded_model_dispatch(pool)
    assert reparked == 0

    # Even a FORCED re-park (the dedup guard is what protects us) writes no second harvest.
    await write_harvest_event(
        pool,
        session_id,
        run_id=inner_run_id,
        outcome="ok",
        output=run_output,
        error=None,
        account_id=_ACCOUNT,
    )
    assert await _harvest_events(pool, session_id) == 1, "harvest write is idempotent on run_id"

    # The fold still produces exactly one assistant turn.
    await run_session_step(session_id, cause="model_workflow_harvest")
    assert len(await _assistant_messages(pool, session_id)) == 1


async def test_harvest_length_finish_reason_preserves_turn(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """A harvested length stop records truncation without crashing the shared tail."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool, finish_reason="length")
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    run_output = await _resolve_inner_run(pool, inner_run_id)
    await write_harvest_event(
        pool,
        session_id,
        run_id=inner_run_id,
        outcome="ok",
        output=run_output,
        error=None,
        account_id=_ACCOUNT,
    )

    await run_session_step(session_id, cause="model_workflow_harvest")

    assistants = await _assistant_messages(pool, session_id)
    assert len(assistants) == 1
    assert assistants[0]["content"] == "recovered answer"

    events = await sessions_service.read_events(pool, session_id, account_id=_ACCOUNT)
    end_spans = [
        e.data for e in events if e.kind == "span" and e.data.get("event") == "model_request_end"
    ]
    assert len(end_spans) == 1
    assert end_spans[0]["finish_reason"] == "length"
    assert end_spans[0]["output_truncated"] is True
    assert end_spans[0]["model"] is not None

    assert not any(e.kind == "span" and e.data.get("event") == "harness_error" for e in events)


async def test_steady_state_park_is_not_double_parked(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """A park whose live harvest task is in-flight in THIS worker is not re-parked."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    # Simulate the live (not-crashed) harvest task: its in-flight key is registered.
    mwf._INFLIGHT_HARVESTS.add((session_id, inner_run_id))
    try:
        # The park IS still unharvested (the live task hasn't written it yet) — so it is a
        # query candidate — but the in-flight gate means the sweep does NOT re-park it.
        async with pool.acquire() as conn:
            stranded = await db_queries.find_unharvested_model_dispatch_parks(conn)
        assert (session_id, inner_run_id, _ACCOUNT) in stranded
        reparked = await repark_stranded_model_dispatch(pool)
        assert reparked == 0, "a park with a live in-flight harvest task must not be double-parked"
    finally:
        mwf._INFLIGHT_HARVESTS.discard((session_id, inner_run_id))


async def test_consumed_park_is_not_reparked(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """A fully-folded (consumed) park is not a crash-recovery candidate."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    run_output = await _resolve_inner_run(pool, inner_run_id)
    await write_harvest_event(
        pool,
        session_id,
        run_id=inner_run_id,
        outcome="ok",
        output=run_output,
        error=None,
        account_id=_ACCOUNT,
    )
    # Fold the harvest — writes the consumed marker (model_workflow_harvest_end).
    await run_session_step(session_id, cause="model_workflow_harvest")
    assert await _harvest_end_events(pool, session_id) == 1

    async with pool.acquire() as conn:
        stranded = await db_queries.find_unharvested_model_dispatch_parks(conn)
    assert session_id not in [s[0] for s in stranded], "a consumed park is never re-parked"
    assert await repark_stranded_model_dispatch(pool) == 0


# ── #2469: a crash between the park record and the run launch ────────────────


async def _bound_ref(pool: asyncpg.Pool[Any], session_id: str) -> WorkflowModelRef:
    async with pool.acquire() as conn:
        model = await conn.fetchval(
            "SELECT a.model FROM sessions s JOIN agents a ON a.id = s.agent_id WHERE s.id = $1",
            session_id,
        )
    ref = parse_workflow_model(model)
    assert ref is not None
    return ref


def _request() -> LlmRequest:
    return LlmRequest(messages=[{"role": "user", "content": "answer this"}])


async def test_crash_after_launch_does_not_launch_a_second_run(
    mwf_runtime: asyncpg.Pool[Any],
) -> None:
    """The park is recorded and its run created, but the step died before spawning the
    harvest task. The next wake must find the park and wait on that run, not launch a
    second paid one."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    real_launch = wf_service.launch_awaited_run

    async def _launch_then_crash(*args: Any, **kwargs: Any) -> Any:
        await real_launch(*args, **kwargs)
        raise RuntimeError("worker died right after the run was created")

    with (
        mock.patch.object(wf_service, "launch_awaited_run", side_effect=_launch_then_crash),
        pytest.raises(RuntimeError),
    ):
        await mwf.launch_model_workflow_park(
            pool,
            session_id,
            ref=await _bound_ref(pool, session_id),
            request=_request(),
            reacting_to=1,
            request_record={},
            account_id=_ACCOUNT,
        )
    await run_session_step(session_id)

    assert len(await _inner_run_ids(pool, session_id)) == 1


async def test_crash_after_record_before_launch_launches_the_recorded_run(
    mwf_runtime: asyncpg.Pool[Any],
) -> None:
    """The park record exists but its run was never created. The next wake launches
    exactly the recorded run id, and the turn then completes normally."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)

    with (
        mock.patch.object(
            wf_service,
            "launch_awaited_run",
            side_effect=RuntimeError("worker died before the run was created"),
        ),
        pytest.raises(RuntimeError),
    ):
        await mwf.launch_model_workflow_park(
            pool,
            session_id,
            ref=await _bound_ref(pool, session_id),
            request=_request(),
            reacting_to=1,
            request_record={},
            account_id=_ACCOUNT,
        )
    async with pool.acquire() as conn:
        park = await db_queries.find_latest_model_workflow_park(
            conn, session_id, account_id=_ACCOUNT
        )
    assert park is not None
    assert await _inner_run_ids(pool, session_id) == []
    # The sweep has no run to harvest: the unlaunched park is the step's to launch.
    async with pool.acquire() as conn:
        stranded = await db_queries.find_unharvested_model_dispatch_parks(conn)
    assert session_id not in [s[0] for s in stranded]

    await run_session_step(session_id)

    assert await _inner_run_ids(pool, session_id) == [park["run_id"]]
    run_output = await _resolve_inner_run(pool, park["run_id"])
    await write_harvest_event(
        pool,
        session_id,
        run_id=park["run_id"],
        outcome="ok",
        output=run_output,
        error=None,
        account_id=_ACCOUNT,
    )
    await run_session_step(session_id, cause="model_workflow_harvest")
    assistants = await _assistant_messages(pool, session_id)
    assert [a["content"] for a in assistants] == ["recovered answer"]


# ── #2470: the account run cap at the park ────────────────────────────────────


async def _span_events(pool: asyncpg.Pool[Any], session_id: str, event: str) -> list[Any]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'span' "
            "AND data->>'event' = $2 ORDER BY seq",
            session_id,
            event,
        )
    return [r["data"] for r in rows]


async def test_account_run_cap_backs_off_instead_of_crashing_the_step(
    mwf_runtime: asyncpg.Pool[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the account's outstanding-run cap the park can't launch. That is capacity,
    not a harness crash: the session records a ``run_capacity`` error and backs off,
    and once capacity frees the next wake launches the recorded run."""
    from aios.config import get_settings

    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    full = get_settings().model_copy(update={"workflow_runs_per_account_max": 0})
    monkeypatch.setattr("aios.workflows.service.get_settings", lambda: full)

    await run_session_step(session_id)

    assert await _span_events(pool, session_id, "harness_error") == []
    [refused] = await _span_events(pool, session_id, "model_workflow_launch_refused")
    assert refused["error"]["kind"] == "run_capacity"
    assert refused["error"]["detail"] == {"outstanding": 0, "max": 0}
    assert await _inner_run_ids(pool, session_id) == []
    session = await sessions_service.get_session(pool, session_id, account_id=_ACCOUNT)
    assert session.stop_reason == {"type": "rescheduling"}

    monkeypatch.undo()
    await run_session_step(session_id)
    async with pool.acquire() as conn:
        park = await db_queries.find_latest_model_workflow_park(
            conn, session_id, account_id=_ACCOUNT
        )
    assert park is not None
    assert await _inner_run_ids(pool, session_id) == [park["run_id"]]


async def test_account_run_cap_exhausts_the_backoff_into_an_errored_turn(
    mwf_runtime: asyncpg.Pool[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap that never frees exhausts the model-error ladder: the session latches
    errored without a ``harness_error`` span and without re-raising out of the step."""
    from aios.config import get_settings

    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    full = get_settings().model_copy(update={"workflow_runs_per_account_max": 0})
    monkeypatch.setattr("aios.workflows.service.get_settings", lambda: full)

    for _ in range(5):
        await run_session_step(session_id)

    assert await _span_events(pool, session_id, "harness_error") == []
    assert len(await _span_events(pool, session_id, "model_workflow_launch_refused")) == 5
    assert await _inner_run_ids(pool, session_id) == []
    session = await sessions_service.get_session(pool, session_id, account_id=_ACCOUNT)
    assert session.stop_reason is not None
    assert session.stop_reason["type"] == "error"
    assert "outstanding-run cap" in session.stop_reason["message"]


async def test_a_permanent_launch_refusal_ends_the_turn_naming_the_cause(
    mwf_runtime: asyncpg.Pool[Any],
) -> None:
    """An archived bound workflow refuses every launch. Retrying can't help, so the
    turn errors at once with the cause named, not via ``harness_error`` retries."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    ref = await _bound_ref(pool, session_id)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE workflows SET archived_at = now() WHERE id = $1", ref.workflow_id
        )

    await run_session_step(session_id)

    assert await _span_events(pool, session_id, "harness_error") == []
    [refused] = await _span_events(pool, session_id, "model_workflow_launch_refused")
    assert refused["error"]["kind"] == "conflict"
    session = await sessions_service.get_session(pool, session_id, account_id=_ACCOUNT)
    assert session.stop_reason is not None
    assert session.stop_reason["type"] == "error"
    assert "archived" in session.stop_reason["message"]


async def test_the_launcher_cap_never_blocks_the_sessions_own_turn(
    mwf_runtime: asyncpg.Pool[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-launcher cap bounds the model's own fan-out. A model-dispatch run is the
    session's turn, so a session at that cap can still run (and stop its runs)."""
    from aios.config import get_settings

    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    full = get_settings().model_copy(update={"workflow_runs_per_launcher_max": 0})
    monkeypatch.setattr("aios.workflows.service.get_settings", lambda: full)

    await run_session_step(session_id)

    assert await _span_events(pool, session_id, "model_workflow_launch_refused") == []
    assert len(await _inner_run_ids(pool, session_id)) == 1


# ── #2471: the park record carries the captured request ───────────────────────


async def test_a_parked_turn_rebuilds_exactly_for_its_capability_model(
    mwf_runtime: asyncpg.Pool[Any],
) -> None:
    """The bound workflow declares an output model, so the turn's capability model
    (which drives the window and the vision and thinking gates) differs from its
    ``workflow:`` model. The park record's request rebuilds byte for byte; rendered
    for another model it reports ``rerendered``."""
    forget_stored_blobs()
    pool = mwf_runtime
    session_id = await _make_bound_session(pool, output_model="openrouter/declared-out")
    await run_session_step(session_id)

    async with pool.acquire() as conn:
        park = await conn.fetchrow(
            "SELECT id, data FROM events WHERE session_id = $1 AND kind = 'span' "
            "AND data->>'event' = 'model_workflow_park'",
            session_id,
        )
    assert park is not None
    record = park["data"]["request"]
    assert record["model"].startswith("workflow:")
    assert record["capability_model"] == "openrouter/declared-out"

    rebuilt = await rebuild_request(
        pool, account_id=_ACCOUNT, session_id=session_id, request_event_id=park["id"]
    )
    assert isinstance(rebuilt, Rebuilt) and rebuilt.fidelity == "exact"
    other = await rebuild_request(
        pool,
        account_id=_ACCOUNT,
        session_id=session_id,
        request_event_id=park["id"],
        target_model="openrouter/other-model",
    )
    assert isinstance(other, Rebuilt) and other.fidelity == "rerendered"
