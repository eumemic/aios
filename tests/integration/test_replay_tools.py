"""The replay run tools (#2475): ``sample_requests``, ``get_request`` and minted-ref grants.

Request spans are seeded straight into ``events`` so the tests control when each was
sent, how it ended, and what its record names. A real rebuild needs a real slate, so
``get_request`` and the by-ref ``call_llm`` are exercised with ``rebuild_request``
stubbed; B1/B2's tests cover the rebuild itself.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.ids import EVENT, make_id
from aios.models.agents import ToolSpec
from aios.models.workflows import OperatorAuthority, RequestRef, SessionAuthority
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.services import workflows as wf_service
from aios.services.requests import Rebuilt
from aios.workflows import run_replay, run_tools, service
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration

_ACC = "acc_replay"
_ENV = "env_replay"
_DAY = datetime(2026, 9, 1, tzinfo=UTC)
_REPLAY_TOOLS = [ToolSpec(type="sample_requests"), ToolSpec(type="get_request")]
_names = itertools.count()
_seqs = itertools.count(1000, 10)


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=6)
    prev = runtime.pool
    runtime.pool = pool
    run_tools._INFLIGHT.clear()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'replay')",
                _ACC,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'replay-env', '{}'::jsonb, $2)",
                _ENV,
                _ACC,
            )
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.run_tools.defer_run_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _agent(pool: asyncpg.Pool[Any]) -> str:
    agent = await agents_service.create_agent(
        pool,
        account_id=_ACC,
        name=f"agent-{next(_names)}",
        model="openrouter/prod",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    return agent.id


async def _session(pool: asyncpg.Pool[Any], agent_id: str) -> str:
    session = await sessions_service.create_session(
        pool, account_id=_ACC, agent_id=agent_id, environment_id=_ENV, title=None, metadata={}
    )
    return session.id


async def _event(
    conn: asyncpg.Connection[Any],
    session_id: str,
    seq: int,
    kind: str,
    data: dict[str, Any],
    at: datetime,
) -> str:
    event_id = make_id(EVENT)
    await conn.execute(
        "INSERT INTO events (id, session_id, seq, kind, data, created_at, account_id) "
        "VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)",
        event_id,
        session_id,
        seq,
        kind,
        json.dumps(data),
        at,
        _ACC,
    )
    return event_id


async def _request(
    pool: asyncpg.Pool[Any],
    session_id: str,
    agent_id: str,
    at: datetime,
    *,
    payload_sha: str | None = None,
    end: str = "ok",
    park: bool = False,
    purpose: str | None = None,
) -> tuple[str, str | None]:
    """Seed one request span and how it ended. Returns ``(span id, answer message id)``.

    ``end`` is ``ok`` (answered), ``error``, ``cancelled`` (no ``model`` on the end),
    ``refused`` (a ``content_filter`` end, no assistant message) or ``none`` (never
    closed)."""
    seq = next(_seqs)
    record = {
        "payload_sha": payload_sha or make_id(EVENT),
        "system_sha": "sha-system",
        "tools_sha": "sha-tools",
        "params_sha": "sha-params",
        "model": "openrouter/prod",
        "capability_model": "openrouter/prod",
        "binding": {"kind": "agent", "agent_id": agent_id, "version": 1},
    }
    async with pool.acquire() as conn:
        run_id = make_id(EVENT)
        start: dict[str, Any] = {
            "event": "model_workflow_park" if park else "model_request_start",
            "request": record,
        }
        if park:
            start["run_id"] = run_id
        if purpose is not None:
            start["purpose"] = purpose
        span_id = await _event(conn, session_id, seq, "span", start, at)
        if end == "none":
            return span_id, None
        if park:
            done: dict[str, Any] = {
                "event": "model_workflow_harvest_end",
                "run_id": run_id,
                "is_error": end == "error",
            }
        else:
            done = {
                "event": "model_request_end",
                "model_request_start_id": span_id,
                "is_error": end == "error",
            }
            if end != "cancelled":
                done["model"] = "openrouter/prod"
            if end == "refused":
                done["finish_reason"] = "content_filter"
        await _event(conn, session_id, seq + 1, "span", done, at + timedelta(seconds=1))
        if end == "refused":
            # A refusal latches an errored turn: no assistant message is persisted.
            return span_id, None
        answer = await _event(
            conn,
            session_id,
            seq + 2,
            "message",
            {"role": "assistant", "content": "answer"},
            at + timedelta(seconds=2),
        )
    return span_id, answer


async def _blobs(pool: asyncpg.Pool[Any], *shas: str) -> None:
    async with pool.acquire() as conn:
        for sha in shas:
            await conn.execute(
                "INSERT INTO request_blobs (account_id, sha256, body) VALUES ($1, $2, $3) "
                "ON CONFLICT DO NOTHING",
                _ACC,
                sha,
                b"{}",
            )


async def _replay_run(
    pool: asyncpg.Pool[Any], script: str = "async def main(input):\n    return 1\n"
) -> Any:
    wf = await wf_service.create_workflow(
        pool,
        account_id=_ACC,
        name=f"eval-{next(_names)}",
        script=script,
        tools=_REPLAY_TOOLS,
    )
    return await service.create_run(
        pool,
        account_id=_ACC,
        authority=OperatorAuthority(),
        workflow_id=wf.id,
        environment_id=_ENV,
    )


async def _sample(run: Any, **args: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "start": _DAY.isoformat(),
        "end": (_DAY + timedelta(days=3)).isoformat(),
        "n": 100,
        "seed": "s1",
        "cluster_cap": 2,
    }
    return await run_replay.invoke_replay_tool(
        run=run, tool_name="sample_requests", args={**base, **args}
    )


# ── sample_requests ───────────────────────────────────────────────────────────


async def test_the_sample_is_answered_deduped_capped_and_seeded(
    pool: asyncpg.Pool[Any],
) -> None:
    await _blobs(pool, "sha-system", "sha-tools", "sha-params")
    agent = await _agent(pool)
    busy = await _session(pool, agent)
    quiet = await _session(pool, agent)
    clone = await _session(pool, agent)
    # Five answered requests in one session on one day: the cluster cap keeps two.
    for i in range(5):
        await _request(pool, busy, agent, _DAY + timedelta(hours=i))
    # A second day of the same session is its own cluster.
    day2, day2_answer = await _request(pool, busy, agent, _DAY + timedelta(days=1))
    # An original and the clone's copy of it: one item, the original.
    original, _ = await _request(pool, quiet, agent, _DAY, payload_sha="sha-shared")
    await _request(pool, clone, agent, _DAY + timedelta(hours=1), payload_sha="sha-shared")
    # A park its run answered counts; nothing else below does.
    park, _ = await _request(pool, quiet, agent, _DAY + timedelta(hours=2), park=True)
    await _request(pool, quiet, agent, _DAY + timedelta(hours=3), end="error")
    await _request(pool, quiet, agent, _DAY + timedelta(hours=4), end="cancelled")
    await _request(pool, quiet, agent, _DAY + timedelta(hours=5), end="none")
    await _request(pool, quiet, agent, _DAY + timedelta(hours=6), park=True, end="error")
    await _request(pool, quiet, agent, _DAY + timedelta(hours=7), purpose="auto_review")
    # Outside the range.
    await _request(pool, quiet, agent, _DAY + timedelta(days=5))
    run = await _replay_run(pool)

    result = await _sample(run, agent_id=agent)

    items = result["items"]
    ids = {i["request_ref"]["request_id"] for i in items}
    assert {original, park, day2} <= ids
    clusters: dict[tuple[str, str], int] = {}
    for item in items:
        key = (item["session_id"], item["created_at"][:10])
        clusters[key] = clusters.get(key, 0) + 1
    assert clusters == {
        (busy, "2026-09-01"): 2,
        (busy, "2026-09-02"): 1,
        (quiet, "2026-09-01"): 2,
    }
    assert len(items) == 5
    day2_item = next(i for i in items if i["request_ref"]["request_id"] == day2)
    assert day2_item["response_event_id"] == day2_answer
    assert day2_item["missing"] is False
    assert day2_item["agent_version"] == 1
    assert day2_item["model"] == "openrouter/prod"

    again = await _sample(run, agent_id=agent)
    assert again == result
    other = await _sample(run, agent_id=agent, seed="s2", n=2)
    first_two = await _sample(run, agent_id=agent, n=2)
    assert other["items"] != first_two["items"]


async def test_a_sampled_request_is_paired_with_its_own_answer(
    pool: asyncpg.Pool[Any],
) -> None:
    """A refused request has no assistant message, so it isn't sampled (its "answer"
    would be the next turn's). A relaunched park's first span was never sent: only the
    last park for the run counts, paired with the harvest's message."""
    agent = await _agent(pool)
    session = await _session(pool, agent)
    await _request(pool, session, agent, _DAY, end="refused")
    later, later_answer = await _request(pool, session, agent, _DAY + timedelta(hours=1))
    run_id = make_id(EVENT)
    at = _DAY + timedelta(hours=2)
    record = {
        "system_sha": "sha-system",
        "tools_sha": "sha-tools",
        "params_sha": "sha-params",
        "model": "workflow:wf_x",
        "capability_model": "openrouter/prod",
        "binding": {"kind": "agent", "agent_id": agent, "version": 1},
    }
    seq = next(_seqs)
    async with pool.acquire() as conn:
        await _event(
            conn,
            session,
            seq,
            "span",
            {
                "event": "model_workflow_park",
                "run_id": run_id,
                "request": {**record, "payload_sha": "sha-unsent"},
            },
            at,
        )
        relaunched = await _event(
            conn,
            session,
            seq + 1,
            "span",
            {
                "event": "model_workflow_park",
                "run_id": run_id,
                "request": {**record, "payload_sha": "sha-sent"},
            },
            at + timedelta(seconds=1),
        )
        await _event(
            conn,
            session,
            seq + 2,
            "span",
            {"event": "model_workflow_harvest_end", "run_id": run_id, "is_error": False},
            at + timedelta(seconds=2),
        )
        harvest_answer = await _event(
            conn,
            session,
            seq + 3,
            "message",
            {"role": "assistant", "content": "deliberated"},
            at + timedelta(seconds=3),
        )
    run = await _replay_run(pool)

    result = await _sample(run, agent_id=agent, cluster_cap=10)

    answers = {i["request_ref"]["request_id"]: i["response_event_id"] for i in result["items"]}
    assert answers == {later: later_answer, relaunched: harvest_answer}


async def test_the_sample_skips_eval_traffic_and_reports_missing_blobs(
    pool: asyncpg.Pool[Any],
) -> None:
    """A session a replay run spawned is eval traffic, not production. A request whose
    blob is gone is reported ``missing`` without a rebuild."""
    await _blobs(pool, "sha-system", "sha-tools")  # no params blob
    agent = await _agent(pool)
    prod = await _session(pool, agent)
    eval_child = await _session(pool, agent)
    run = await _replay_run(pool)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET parent_run_id = $1, agent_version = 1 WHERE id = $2",
            run.id,
            eval_child,
        )
    prod_span, _ = await _request(pool, prod, agent, _DAY)
    await _request(pool, eval_child, agent, _DAY)

    result = await _sample(run, agent_id=agent)

    assert [i["request_ref"]["request_id"] for i in result["items"]] == [prod_span]
    assert result["items"][0]["missing"] is True


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"end": _DAY.isoformat()}, "start must be before end"),
        ({"end": (_DAY + timedelta(days=40)).isoformat()}, "at most 31 days"),
        ({"n": 0}, "bad arguments"),
        ({"start": "2026-09-01T00:00:00"}, "bad arguments"),  # no timezone
        ({"extra": 1}, "bad arguments"),
    ],
)
async def test_a_bad_sample_is_an_error_value(
    pool: asyncpg.Pool[Any], args: dict[str, Any], message: str
) -> None:
    run = await _replay_run(pool)
    result = await _sample(run, agent_id="agt_x", **args)
    assert message in result["error"]


# ── grants ────────────────────────────────────────────────────────────────────


async def _journal_call(
    pool: asyncpg.Pool[Any], run_id: str, key: str, started: dict[str, Any], result: Any
) -> None:
    async with pool.acquire() as conn:
        await wf_queries.append_run_event(
            conn, account_id=_ACC, run_id=run_id, type="call_started", call_key=key, payload=started
        )
        await wf_queries.append_run_event(
            conn,
            account_id=_ACC,
            run_id=run_id,
            type="call_result",
            call_key=key,
            payload={"result": result, "is_error": False},
        )


async def test_only_a_sample_requests_result_grants_its_refs(pool: asyncpg.Pool[Any]) -> None:
    """A ref is granted by the call that produced it: a ``sample_requests`` tool call.
    The same ref-shaped value in a gate resume or another tool's result grants nothing."""
    run = await _replay_run(pool)
    minted = {"session_id": "ses_a", "request_id": "evt_a"}
    gated = {"session_id": "ses_b", "request_id": "evt_b"}
    fetched = {"session_id": "ses_c", "request_id": "evt_c"}
    async with pool.acquire() as conn:
        await wf_queries.append_run_event(
            conn, account_id=_ACC, run_id=run.id, type="run_started", payload={"input": None}
        )
    await _journal_call(
        pool,
        run.id,
        "k1",
        {"capability": "tool", "tool_name": "sample_requests", "input": {}},
        {"items": [{"request_ref": minted}]},
    )
    await _journal_call(
        pool, run.id, "k2", {"capability": "gate"}, {"items": [{"request_ref": gated}]}
    )
    await _journal_call(
        pool,
        run.id,
        "k3",
        {"capability": "tool", "tool_name": "web_fetch", "input": {}},
        {"items": [{"request_ref": fetched}]},
    )

    async with pool.acquire() as conn:
        assert await run_replay.request_ref_granted(conn, run, minted)
        assert await run_replay.request_ref_granted(conn, run, {**minted})
        assert not await run_replay.request_ref_granted(conn, run, gated)
        assert not await run_replay.request_ref_granted(conn, run, fetched)
        assert not await run_replay.request_ref_granted(conn, run, {**minted, "x": "y"})
        assert not await run_replay.request_ref_granted(conn, run, "evt_a")


async def _drive(run_id: str, wakes: int = 6) -> None:
    """Step the run, letting each wake's tool tasks finish before the next."""
    for _ in range(wakes):
        await run_workflow_step(run_id)
        tasks = list(run_tools._INFLIGHT.values())
        if tasks:
            await asyncio.gather(*tasks)


_SAMPLE_THEN_USE = """
async def main(input):
    sample = await tool('sample_requests', input['sample'])
    ref = sample['items'][0]['request_ref']
    got = await tool('get_request', {'request_ref': ref, 'model': 'openrouter/judge'})
    sent = await call_llm(request_ref=ref, model='openrouter/judge')
    return {'got': got, 'sent': sent, 'ref': ref}
"""


async def test_an_eval_samples_then_reads_and_sends_a_minted_ref(
    pool: asyncpg.Pool[Any],
) -> None:
    await _blobs(pool, "sha-system", "sha-tools", "sha-params")
    agent = await _agent(pool)
    session = await _session(pool, agent)
    span, _ = await _request(pool, session, agent, _DAY)
    wf = await wf_service.create_workflow(
        pool,
        account_id=_ACC,
        name="eval-e2e",
        script=_SAMPLE_THEN_USE,
        tools=_REPLAY_TOOLS,
    )
    run = await service.create_run(
        pool,
        account_id=_ACC,
        authority=OperatorAuthority(),
        workflow_id=wf.id,
        environment_id=_ENV,
        input={
            "sample": {
                "agent_id": agent,
                "start": _DAY.isoformat(),
                "end": (_DAY + timedelta(days=1)).isoformat(),
                "n": 10,
                "seed": 7,
            }
        },
    )
    rebuilt = Rebuilt(
        request={"messages": [{"role": "user", "content": "q"}], "tools": None, "params": {}},
        fidelity="rerendered",
        record={"model": "openrouter/prod", "capability_model": "openrouter/prod"},
    )
    with (
        mock.patch("aios.workflows.run_replay.rebuild_request", AsyncMock(return_value=rebuilt)),
        mock.patch("aios.workflows.step.run_llm.launch_call_llm_task") as launch,
    ):
        await _drive(run.id)

    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run.id)
    results = [e.payload["result"] for e in events if e.type == "call_result"]
    assert results[0]["items"][0]["request_ref"] == {"session_id": session, "request_id": span}
    assert results[1] == {
        "messages": [{"role": "user", "content": "q"}],
        "tools": None,
        "params": None,  # another model gets no captured params
        "fidelity": "rerendered",
    }
    # Granted: the by-ref call is launched (re-dispatched on each wake here, since the
    # stubbed launch never signals), never refused.
    assert launch.called
    assert not [r for r in results if isinstance(r, dict) and "error_kind" in r]
    assert launch.call_args.kwargs["spec"]["request_ref"] == {
        "session_id": session,
        "request_id": span,
    }


async def test_a_minted_ref_can_be_handed_to_a_sub_run(pool: asyncpg.Pool[Any]) -> None:
    await _blobs(pool, "sha-system", "sha-tools", "sha-params")
    agent = await _agent(pool)
    session = await _session(pool, agent)
    span, _ = await _request(pool, session, agent, _DAY)
    async with pool.acquire() as conn:
        arm = await wf_queries.insert_workflow(
            conn, account_id=_ACC, name="arm", script="async def main(input):\n    return 1\n"
        )
    wf = await wf_service.create_workflow(
        pool,
        account_id=_ACC,
        name="eval-hand-on",
        script=(
            "async def main(input):\n"
            "    sample = await tool('sample_requests', input['sample'])\n"
            "    ref = sample['items'][0]['request_ref']\n"
            "    return await invoke_workflow(input['arm'], None, request_ref=ref)\n"
        ),
        tools=_REPLAY_TOOLS,
    )
    run = await service.create_run(
        pool,
        account_id=_ACC,
        authority=OperatorAuthority(),
        workflow_id=wf.id,
        environment_id=_ENV,
        input={
            "arm": arm.id,
            "sample": {
                "agent_id": agent,
                "start": _DAY.isoformat(),
                "end": (_DAY + timedelta(days=1)).isoformat(),
                "n": 1,
                "seed": 1,
            },
        },
    )

    await _drive(run.id, 3)

    async with pool.acquire() as conn:
        sub_id = await conn.fetchval("SELECT id FROM wf_runs WHERE parent_run_id = $1", run.id)
        sub = await wf_queries.get_run_for_step(conn, sub_id)
    assert sub is not None
    assert sub.request_ref == RequestRef(session_id=session, request_id=span)


async def test_a_session_run_cannot_sample(pool: asyncpg.Pool[Any]) -> None:
    """End to end: a session launch of a replay workflow drops the tools (the clamp),
    so its sample call is refused as a value and reads nothing."""
    agent = await _agent(pool)
    session = await _session(pool, agent)
    wf = await wf_service.create_workflow(
        pool,
        account_id=_ACC,
        name="eval-session",
        script=(
            "async def main(input):\n"
            "    return await tool('sample_requests', {'agent_id': 'x', 'start': '', "
            "'end': '', 'n': 1, 'seed': 1})\n"
        ),
        tools=_REPLAY_TOOLS,
    )
    run = await service.create_run(
        pool,
        account_id=_ACC,
        authority=SessionAuthority(session, None),
        workflow_id=wf.id,
        environment_id=_ENV,
    )
    with mock.patch(
        "aios.workflows.run_replay.request_queries.sample_request_spans"
    ) as sample_query:
        await _drive(run.id, 3)
    sample_query.assert_not_called()
    async with pool.acquire() as conn:
        done = await wf_queries.get_run_for_step(conn, run.id)
    assert done is not None and "only callable from an operator run" in done.output["error"]
