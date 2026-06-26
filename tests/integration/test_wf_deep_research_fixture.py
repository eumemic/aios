"""Deep-research reference workflow fixture (#792).

The live demo runs the width-10 script against real web-capable agents.  This CI
fixture drives the same workflow shape directly against Postgres with a tiny fan-out
and simulated child returns: sweep -> restart/replay -> deep read pipeline ->
synthesis -> critic -> complete.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.workflows import WfRunEvent
from aios.services import agents as agents_service
from aios.tools import workflow_completion
from aios.workflows import run_tools, service
from aios.workflows.deep_research import build_deep_research_fixture_script
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration


@pytest.fixture
async def wf_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_wf', NULL, TRUE, 'wf-root')"
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_wf', 'wf-env', '{}'::jsonb, 'acc_wf')"
            )
        run_tools._INFLIGHT.clear()
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
            mock.patch("aios.workflows.run_tools.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.sessions.defer_run_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _make_run(pool: asyncpg.Pool[Any], script: str, *, input: Any = None) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(conn, account_id="acc_wf", name="w", script=script)
    run = await service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf", input=input
    )
    return run.id


async def _open_request_id(pool: asyncpg.Pool[Any], session_id: str) -> str:
    async with pool.acquire() as conn:
        ids = await db_queries.get_open_request_ids(conn, session_id, account_id="acc_wf")
    return ids[0]


@pytest.fixture
async def deep_research_agents(wf_runtime: asyncpg.Pool[Any]) -> dict[str, str]:
    ids: dict[str, str] = {}
    for role in ("scout", "reader", "synthesis", "critic"):
        agent = await agents_service.create_agent(
            wf_runtime,
            account_id="acc_wf",
            name=f"deep-research-{role}",
            model="test/dummy",
            system=f"test {role}",
            tools=[],
            description=None,
            metadata={},
            window_min=1000,
            window_max=100000,
        )
        ids[role] = agent.id
    return ids


async def _events(pool: asyncpg.Pool[Any], run_id: str) -> list[WfRunEvent]:
    async with pool.acquire() as conn:
        return await wf_queries.list_run_events(conn, run_id)


async def _children_by_payload(pool: asyncpg.Pool[Any], run_id: str) -> dict[str, str]:
    children: dict[str, str] = {}
    for event in await _events(pool, run_id):
        if event.type != "call_started" or event.payload.get("capability") != "agent":
            continue
        child_id = event.payload["child_session_id"]
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM events WHERE session_id = $1 AND account_id = $2 "
                "AND kind = 'message' AND role = 'user' ORDER BY seq ASC LIMIT 1",
                child_id,
                "acc_wf",
            )
        request = row["data"]
        children[request["content"]] = child_id
    return children


async def _return_child(pool: asyncpg.Pool[Any], child_id: str, value: Any) -> None:
    request_id = await _open_request_id(pool, child_id)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        result = await workflow_completion.return_handler(
            child_id, {"request_id": request_id, "value": value}
        )
    assert result == {"status": "returned"}


async def test_deep_research_fixture_survives_replay_and_completes(
    wf_runtime: asyncpg.Pool[Any], deep_research_agents: dict[str, str]
) -> None:
    pool = wf_runtime
    script = build_deep_research_fixture_script(
        scout_agent_id=deep_research_agents["scout"],
        reader_agent_id=deep_research_agents["reader"],
        synthesis_agent_id=deep_research_agents["synthesis"],
        critic_agent_id=deep_research_agents["critic"],
    )
    run_id = await _make_run(pool, script, input={"question": "What changed in AIOS?"})

    await run_workflow_step(run_id)  # Phase 1: two scouts + one catchable missing scout
    events_after_spawn = await _events(pool, run_id)
    assert [
        e.payload["text"]
        for e in events_after_spawn
        if e.type == "annotation" and e.payload["kind"] == "phase"
    ] == ["Phase 1: sweep"]
    assert len([e for e in events_after_spawn if e.type == "call_started"]) == 2
    assert len([e for e in events_after_spawn if e.type == "frontier_deferred"]) == 0
    assert (
        len(
            [e for e in events_after_spawn if e.type == "call_result" and e.payload.get("is_error")]
        )
        == 1
    )

    # Simulated restart/replay before all Phase-1 children return: no duplicate starts
    # and the expected AgentError catch log is emitted exactly once.
    children = await _children_by_payload(pool, run_id)
    first_scout = next(
        cid for content, cid in children.items() if '"angle": "direct-factual"' in content
    )
    await _return_child(
        pool,
        first_scout,
        {
            "angle": "direct-factual",
            "findings": [
                {
                    "claim": "AIOS has a durable workflow runtime.",
                    "source_url": "https://example.com/a?b=2&a=1",
                    "source_title": "AIOS A",
                    "confidence": 0.8,
                }
            ],
            "dead_ends": [],
        },
    )
    await run_workflow_step(run_id)
    replay_events = await _events(pool, run_id)
    assert len([e for e in replay_events if e.type == "call_started"]) == 2
    assert (
        len(
            [
                e
                for e in replay_events
                if e.type == "annotation"
                and e.payload["kind"] == "log"
                and "scout failed (expected)" in e.payload["text"]
            ]
        )
        == 1
    )

    children = await _children_by_payload(pool, run_id)
    second_scout = next(
        cid
        for content, cid in children.items()
        if '"angle": "contrarian-counter-evidence"' in content
    )
    await _return_child(
        pool,
        second_scout,
        {
            "angle": "contrarian-counter-evidence",
            "findings": [
                {
                    "claim": "AIOS has a durable workflow runtime.",
                    "source_url": "https://www.example.com/a/?a=1&b=2",
                    "source_title": "Duplicate AIOS A",
                    "confidence": 0.7,
                },
                {
                    "claim": "Workflow scripts can fan out to agents.",
                    "source_url": "https://example.com/b",
                    "source_title": "AIOS B",
                    "confidence": 0.6,
                },
            ],
            "dead_ends": [],
        },
    )
    await run_workflow_step(run_id)  # Phase 2 reader pipeline spawns two readers
    phase2_events = await _events(pool, run_id)
    assert [
        e.payload["text"]
        for e in phase2_events
        if e.type == "annotation" and e.payload["kind"] == "phase"
    ] == [
        "Phase 1: sweep",
        "Phase 2: deep read",
    ]
    assert len([e for e in phase2_events if e.type == "call_started"]) == 4

    children = await _children_by_payload(pool, run_id)
    reader_children = [cid for content, cid in children.items() if '"source"' in content]
    assert len(reader_children) == 2
    await _return_child(
        pool,
        reader_children[0],
        {
            "source_url": "https://example.com/a?a=1&b=2",
            "credibility": 0.9,
            "key_facts": [{"fact": "Runtime is durable", "quote": "durable", "confidence": 0.8}],
        },
    )
    await _return_child(
        pool,
        reader_children[1],
        {
            "source_url": "https://example.com/b",
            "credibility": 0.7,
            "key_facts": [{"fact": "Scripts fan out", "confidence": 0.7}],
        },
    )
    await run_workflow_step(run_id)  # synthesis

    children = await _children_by_payload(pool, run_id)
    synthesis_child = next(cid for content, cid in children.items() if '"readings"' in content)
    await _return_child(
        pool,
        synthesis_child,
        {
            "report_markdown": "# Report\nAIOS workflow runtime works [1].",
            "citations": [{"n": 1, "url": "https://example.com/a?a=1&b=2", "title": "AIOS A"}],
            "open_questions": [],
        },
    )
    await run_workflow_step(run_id)  # critic

    children = await _children_by_payload(pool, run_id)
    critic_child = next(cid for content, cid in children.items() if '"draft"' in content)
    await _return_child(
        pool,
        critic_child,
        {"verdict": "complete", "gaps": []},
    )
    await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output["report"].startswith("# Report")
    assert run.output["citations"] == [
        {"n": 1, "url": "https://example.com/a?a=1&b=2", "title": "AIOS A"}
    ]
    assert run.output["critic_verdict"] == {"verdict": "complete", "gaps": []}
    assert run.output["stats"] == {
        "scouts": 2,
        "sources": 2,
        "readers": 2,
        "supplementary_scouts": 0,
    }

    final_events = await _events(pool, run_id)
    assert [
        e.payload["text"]
        for e in final_events
        if e.type == "annotation" and e.payload["kind"] == "phase"
    ] == [
        "Phase 1: sweep",
        "Phase 2: deep read",
        "Phase 3: synthesis",
    ]
    call_keys = [e.call_key for e in final_events if e.type == "call_started"]
    assert len(call_keys) == len(set(call_keys)) == 6
    result_keys = [
        e.call_key
        for e in final_events
        if e.type == "call_result" and not e.payload.get("is_error")
    ]
    assert len(result_keys) == len(set(result_keys)) == 6
