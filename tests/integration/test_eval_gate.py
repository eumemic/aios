"""The WaM gate end to end, against fake models.

Each sampled session hides a code (``CTX:<code>``) early in its conversation and ends by
asking for it. A good model finds the code in its context and answers ``ANS:<code>``;
a weak model, or one shown too little context, answers ``ANS:?``. The fake judge
prefers the reply that names the code when the code is in the part of the
conversation it is shown, and otherwise always picks the first reply it sees (a
position bias), so its two orders disagree and every comparison is a tie.

This checks the gate's acceptance criteria with real workflow runs, the real script
host and the registered templates: R0 against the same model is PASS, a weaker model
is FAIL, and a judge shown too little context is INVALID because the negative control
isn't shown worse. Then the stopping rules: too few items for power, a judge sharing a
family with an arm, a spent budget, and an analysis that fails.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest
from evals.workflows import eval_analysis, eval_item, eval_judge, eval_r0, paired_eval

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.harness.completion import LlmRequest, LlmResponse
from aios.ids import EVENT, make_id
from aios.models.agents import ToolSpec
from aios.models.workflows import TERMINAL_RUN_STATUSES, OperatorAuthority, WfRun
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.services import workflows as wf_service
from aios.services.requests import Rebuilt
from aios.workflows import run_llm, run_tools, service
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration

_ACC = "acc_evalgate"
_ENV = "env_evalgate"
_DAY = datetime(2026, 9, 1, tzinfo=UTC)
GOOD = "openai/gpt-4o"
WEAK = "openai/gpt-4o-mini"
JUDGE = "anthropic/claude-3-5-sonnet-20240620"
_SESSIONS = 16
_BAR: dict[str, Any] = json.loads(
    (Path(__file__).parents[2] / "evals" / "bars" / "wam_gate.json").read_text()
)


def _bar(**changes: Any) -> dict[str, Any]:
    """The default bar, scaled down to a 16-session corpus."""
    bar = copy.deepcopy(_BAR)
    bar.update(
        sample_size=20,
        cluster_cap=1,
        min_clusters=12,
        power=0.01,
        wave_size=4,
        max_attempts=2,
        bootstrap_rounds=200,
    )
    bar["limits"].update(degenerate=0.5, tool_calls=0.5, cost=1.0, latency=1000.0)
    bar["control"]["min_clusters"] = 12
    bar["judge"].update(model=JUDGE, tail=5)
    bar["budget"].update(arm_usd=0.5, candidate_usd=0.5, judge_usd=0.5)
    for key, value in changes.items():
        if isinstance(value, dict):
            bar[key].update(value)
        else:
            bar[key] = value
    return bar


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=20)
    prev = runtime.pool
    runtime.pool = pool
    run_tools._INFLIGHT.clear()
    run_llm._INFLIGHT.clear()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'evalgate')",
                _ACC,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'evalgate-env', '{}'::jsonb, $2)",
                _ENV,
                _ACC,
            )
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.run_tools.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.run_llm.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        run_llm._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


# ── the corpus ────────────────────────────────────────────────────────────────


def _conversation(code: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You remember codes."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": f"CTX:{code}"},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": "filler"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "What was the code?"},
    ]


async def _corpus(pool: asyncpg.Pool[Any]) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """An agent and one answered request per session, each on its own day. Returns the
    agent and each session's conversation."""
    agent = await agents_service.create_agent(
        pool,
        account_id=_ACC,
        name="eval-target",
        model=GOOD,
        system="You remember codes.",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    conversations: dict[str, list[dict[str, Any]]] = {}
    async with pool.acquire() as conn:
        for sha in ("sha-system", "sha-tools", "sha-params"):
            await conn.execute(
                "INSERT INTO request_blobs (account_id, sha256, body) VALUES ($1, $2, $3)",
                _ACC,
                sha,
                b"{}",
            )
    for i in range(_SESSIONS):
        session = await sessions_service.create_session(
            pool, account_id=_ACC, agent_id=agent.id, environment_id=_ENV, title=None, metadata={}
        )
        conversations[session.id] = _conversation(f"c{i}x")
        at = _DAY + timedelta(days=i)
        record = {
            "payload_sha": f"payload-{i}",
            "system_sha": "sha-system",
            "tools_sha": "sha-tools",
            "params_sha": "sha-params",
            "model": GOOD,
            "capability_model": GOOD,
            "binding": {"kind": "agent", "agent_id": agent.id, "version": 1},
        }
        async with pool.acquire() as conn:
            span = await _event(
                conn, session.id, 100, {"event": "model_request_start", "request": record}, at
            )
            await _event(
                conn,
                session.id,
                101,
                {
                    "event": "model_request_end",
                    "model_request_start_id": span,
                    "is_error": False,
                    "model": GOOD,
                },
                at + timedelta(seconds=1),
                kind="span",
            )
            await _event(
                conn,
                session.id,
                102,
                {"role": "assistant", "content": "answer"},
                at + timedelta(seconds=2),
                kind="message",
            )
    return agent.id, conversations


async def _event(
    conn: asyncpg.Connection[Any],
    session_id: str,
    seq: int,
    data: dict[str, Any],
    at: datetime,
    *,
    kind: str = "span",
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


# ── fake models ───────────────────────────────────────────────────────────────


class FakeModels:
    def __init__(
        self, conversations: dict[str, list[dict[str, Any]]], cost: float, overloads: int
    ) -> None:
        self.conversations = conversations
        self.cost = cost
        self.overloads = overloads  # the first arm calls a provider rejects as overloaded
        self.calls: list[str] = []

    async def rebuild(self, pool: Any, **kwargs: Any) -> Rebuilt:
        return Rebuilt(
            request={
                "messages": self.conversations[kwargs["session_id"]],
                "tools": None,
                "params": {},
            },
            fidelity="exact",
            record={"model": GOOD, "capability_model": GOOD},
        )

    async def call(self, request: LlmRequest, *, model: str, auth: Any = None) -> LlmResponse:
        self.calls.append(model)
        prompt = str(request.messages[-1].get("content"))
        if "Which reply is the better next assistant turn?" in prompt:
            content = self._judge(prompt)
        elif self.overloads:
            self.overloads -= 1
            raise RuntimeError("RateLimitError: 429 Too Many Requests")
        elif model == GOOD:
            codes = re.findall(r"CTX:(\w+)", json.dumps(request.messages))
            content = f"ANS:{codes[-1]}" if codes else "ANS:?"
        else:
            content = "ANS:?"
        return LlmResponse(
            content=content,
            tool_calls=[],
            finish_reason="stop",
            usage={"input_tokens": 1000, "output_tokens": 20},
            cost=self.cost,
            message={"role": "assistant", "content": content},
        )

    @staticmethod
    def _judge(prompt: str) -> str:
        conversation = prompt.split("CONVERSATION (most recent part):")[1].split("REPLY 1:")[0]
        reply_1 = prompt.split("REPLY 1:")[1].split("REPLY 2:")[0]
        reply_2 = prompt.split("REPLY 2:")[1]
        codes = re.findall(r"CTX:(\w+)", conversation)
        if not codes:
            return "I can't tell.\nWINNER: 1"
        answer = f"ANS:{codes[-1]}"
        if (answer in reply_1) == (answer in reply_2):
            return "WINNER: TIE"
        return "WINNER: 1" if answer in reply_1 else "WINNER: 2"

    def judge_calls(self) -> int:
        return sum(1 for m in self.calls if m not in (GOOD, WEAK))


# ── registering and driving ───────────────────────────────────────────────────


async def _workflow(
    pool: asyncpg.Pool[Any], name: str, script: str, tools: list[dict[str, str]]
) -> dict[str, Any]:
    wf = await wf_service.create_workflow(
        pool,
        account_id=_ACC,
        name=name,
        script=script,
        tools=[ToolSpec(type=t["type"]) for t in tools],
    )
    return {"id": wf.id, "version": wf.version}


_CANDIDATE = """
async def main(input):
    return await call_llm({
        "model": %r,
        "messages": input["messages"],
        "tools": input.get("tools"),
        "params": input.get("params"),
        "session_id": input.get("session_id"),
    })
"""

_BROKEN_ANALYSIS = """
async def main(input):
    if input["mode"] == "power":
        return {"n": 12, "power": 1.0}
    raise ValueError("a bug in the statistics")
"""


async def _register(
    pool: asyncpg.Pool[Any], bar: dict[str, Any], *, analysis_script: str | None = None
) -> dict[str, Any]:
    r0 = await _workflow(pool, "eval-r0", eval_r0.build(), eval_r0.TOOLS)
    judge = await _workflow(pool, "eval-judge", eval_judge.build(), eval_judge.TOOLS)
    analysis = await _workflow(
        pool, "eval-analysis", analysis_script or eval_analysis.build(), eval_analysis.TOOLS
    )
    item = await _workflow(pool, "eval-item", eval_item.build(r0=r0, judge=judge), eval_item.TOOLS)
    return await _workflow(
        pool,
        "wam-gate",
        paired_eval.build(bar=bar, item=item, analysis=analysis),
        paired_eval.TOOLS,
    )


async def _needing(pool: asyncpg.Pool[Any]) -> list[str]:
    async with pool.acquire() as conn:
        return await wf_queries.list_run_ids_needing_step(
            conn,
            agent_deadline_seconds=3600,
            agent_cost_ceiling_microusd=0,
            tool_stale_seconds=3600,
            bash_default_timeout_seconds=120,
            sandbox_provisioning_slack_seconds=180,
            max_bash_timeout_seconds=3_155_760_000,
            call_llm_stale_seconds=3600,
        )


async def _settle(pool: asyncpg.Pool[Any], run_id: str) -> WfRun:
    """Step every run that has something to do, let its tool and model calls finish,
    and repeat until the gate run ends."""
    limit = asyncio.Semaphore(8)

    async def step(rid: str) -> None:
        async with limit:
            await run_workflow_step(rid)

    for _ in range(300):
        async with pool.acquire() as conn:
            run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        if run.status in TERMINAL_RUN_STATUSES:
            return run
        await asyncio.gather(*(step(rid) for rid in await _needing(pool)))
        tasks = [*run_llm._INFLIGHT.values(), *run_tools._INFLIGHT.values()]
        if tasks:
            await asyncio.gather(*tasks)
    raise AssertionError("the gate run never finished")


async def _gate(
    pool: asyncpg.Pool[Any],
    bar: dict[str, Any],
    *,
    candidate_model: str = GOOD,
    cost: float = 0.001,
    budget_usd: float = 1000.0,
    analysis_script: str | None = None,
    overloads: int = 0,
) -> tuple[dict[str, Any], FakeModels]:
    agent, conversations = await _corpus(pool)
    gate = await _register(pool, bar, analysis_script=analysis_script)
    candidate = await _workflow(pool, "candidate", _CANDIDATE % candidate_model, [])
    models = FakeModels(conversations, cost, overloads)
    run = await service.create_run(
        pool,
        account_id=_ACC,
        authority=OperatorAuthority(),
        workflow_id=gate["id"],
        environment_id=_ENV,
        budget_usd=budget_usd,
        input={
            "agent": {"agent_id": agent, "version": 1},
            "baseline_model": GOOD,
            "candidate": {"workflow_id": candidate["id"], "version": candidate["version"]},
            "seed": "s1",
            "window": {
                "start": _DAY.isoformat(),
                "end": (_DAY + timedelta(days=_SESSIONS)).isoformat(),
            },
            "candidate_created_at": _DAY.isoformat(),
        },
    )
    with (
        mock.patch("aios.workflows.run_replay.rebuild_request", models.rebuild),
        mock.patch("aios.workflows.run_llm.rebuild_request", models.rebuild),
        mock.patch("aios.workflows.step.rebuild_request", models.rebuild),
        mock.patch("aios.workflows.run_llm.call_litellm", models.call),
        mock.patch("aios.workflows.run_llm.runtime.require_crypto_box"),
        mock.patch(
            "aios.workflows.run_llm.model_providers_service.resolve_provider_auth_or_conflict",
            AsyncMock(return_value=(object(), None)),
        ),
    ):
        done = await _settle(pool, run.id)
    assert done.status == "completed", done.output
    return done.output, models


# ── acceptance ────────────────────────────────────────────────────────────────


async def test_r0_against_the_same_model_passes(pool: asyncpg.Pool[Any]) -> None:
    out, _ = await _gate(pool, _bar())
    assert out["verdict"] == "PASS", out["reasons"]
    assert out["stats"]["w"]["value"] == 0.5
    assert out["diagnostics"]["agreement"] == 1.0
    # The judge saw the code, so the control (no code in its context) lost every pair.
    assert out["stats"]["control"]["w"]["value"] == 0.0
    assert out["stats"]["n"] == 12 and out["stats"]["clusters"] == 12
    record = out["records"][0]
    assert record["control_eligible"] is True
    assert record["candidate_resolved"]["models"] == [GOOD]
    assert record["judge_models"] == [JUDGE]
    assert out["candidate_resolved"]["models"] == [GOOD]
    assert out["stats"]["cost"]["value"] == 1.0


async def test_a_weaker_candidate_fails(pool: asyncpg.Pool[Any]) -> None:
    out, _ = await _gate(pool, _bar(), candidate_model=WEAK)
    assert out["verdict"] == "FAIL", out["reasons"]
    assert out["reasons"]["failed"] == ["win_rate"]
    assert out["stats"]["w"]["value"] == 0.0


async def test_a_judge_shown_too_little_context_is_invalid(pool: asyncpg.Pool[Any]) -> None:
    """The judge's tail misses the code: it can't tell the control from the baseline,
    so the control isn't shown worse and the gate is INVALID, even though the
    candidate is the same model."""
    out, _ = await _gate(pool, _bar(judge={"tail": 3}))
    assert out["verdict"] == "INVALID"
    assert out["reasons"]["invalid"] == ["control_not_worse", "control_ties"]
    assert out["stats"]["control"]["w"]["value"] == 0.5


# ── stopping rules ────────────────────────────────────────────────────────────


async def test_too_few_items_for_power_spends_nothing(pool: asyncpg.Pool[Any]) -> None:
    out, models = await _gate(pool, _bar(min_clusters=18))
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["reasons"]["inconclusive"] == ["underpowered"]
    assert out["sample"]["n_required"] == 18 and out["sample"]["eligible"] == _SESSIONS
    assert models.calls == []


async def test_a_judge_sharing_an_arms_family_stops_after_the_probe(
    pool: asyncpg.Pool[Any],
) -> None:
    out, models = await _gate(pool, _bar(judge={"model": "openai/gpt-4o-mini"}))
    assert out["verdict"] == "INVALID"
    assert "judge_family" in out["reasons"]["invalid"]
    assert len(out["records"]) == 1
    assert models.judge_calls() == 0


async def test_a_spent_budget_stops_the_run(pool: asyncpg.Pool[Any]) -> None:
    out, _ = await _gate(pool, _bar(), cost=0.4, budget_usd=3.0)
    assert out["verdict"] == "INCONCLUSIVE"
    assert "budget_stop" in out["reasons"]["inconclusive"]
    assert out["flags"] == ["budget_stop"]
    assert len(out["records"]) == 1


async def test_an_overload_on_any_arm_reruns_the_whole_item(pool: asyncpg.Pool[Any]) -> None:
    """Load the eval causes is never a loss for one arm: the item runs again."""
    out, _ = await _gate(pool, _bar(), overloads=1)
    assert out["verdict"] == "PASS", out["reasons"]
    assert out["exclusions"] == {}
    async with pool.acquire() as conn:
        items = await conn.fetchval(
            "SELECT count(*) FROM wf_runs r JOIN workflows w ON w.id = r.workflow_id "
            "WHERE w.name = 'eval-item'"
        )
    assert items == 13  # the probe ran twice


async def test_a_failed_analysis_keeps_the_records(pool: asyncpg.Pool[Any]) -> None:
    out, _ = await _gate(pool, _bar(), analysis_script=_BROKEN_ANALYSIS)
    assert out["verdict"] is None
    assert "a bug in the statistics" in out["analysis_error"]
    assert len(out["records"]) == 12
    assert all(r["outcomes"]["cand"] == "tie" for r in out["records"])
