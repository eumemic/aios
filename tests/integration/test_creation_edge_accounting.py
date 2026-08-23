"""Issue #2151 acceptance tests for creation-edge usage accounting."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import accounting as accounting_queries
from aios.db.queries import workflows as wf_queries
from aios.models.accounting import UsageNodeRef
from aios.services import sessions as sessions_service
from aios.services import workflows as workflows_service
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

ACCOUNT = "acc_creation_accounting"


@pytest.fixture
async def accounting_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'creation accounting')",
                ACCOUNT,
            )
        yield pool
    finally:
        await pool.close()


async def _seed_agent_workflow_agent_chain(
    pool: asyncpg.Pool[Any],
) -> tuple[str, str, str, str, str]:
    """Return ``(root_session, run, child_session, agent, environment)``."""
    agent, environment, root = await seed_agent_env_session(
        pool, account_id=ACCOUNT, prefix="creation-accounting"
    )
    async with pool.acquire() as conn:
        run = await wf_queries.insert_wf_run(
            conn,
            account_id=ACCOUNT,
            workflow_id=None,
            environment_id=environment.id,
            launcher_session_id=root.id,
            caller={"kind": "session", "id": root.id, "awaited": True},
            script="async def main(input):\n    return input\n",
            script_sha="creation-accounting",
            host_semantics_epoch=HOST_SEMANTICS_EPOCH,
            depth=10,
        )
        child = await queries.insert_child_session(
            conn,
            session_id="ses_creation_child",
            account_id=ACCOUNT,
            agent_id=agent.id,
            environment_id=environment.id,
            agent_version=agent.version,
            model="openrouter/test",
            parent_run_id=run.id,
            tools=[],
            mcp_servers=[],
            http_servers=[],
        )
        assert child is not None
    return root.id, run.id, child.id, agent.id, environment.id


async def _charge_known_chain(
    pool: asyncpg.Pool[Any], root_id: str, run_id: str, child_id: str
) -> None:
    async with pool.acquire() as conn:
        await queries.increment_session_usage(
            conn,
            root_id,
            account_id=ACCOUNT,
            input_tokens=10,
            output_tokens=1,
            cost_microusd=100,
        )
        await wf_queries.add_run_call_llm_cost_microusd(
            conn,
            run_id,
            200,
            account_id=ACCOUNT,
            input_tokens=20,
            output_tokens=2,
        )
        await queries.increment_session_usage(
            conn,
            child_id,
            account_id=ACCOUNT,
            input_tokens=30,
            output_tokens=3,
            cost_microusd=300,
        )


async def test_agent_workflow_agent_rolls_up_exact_own_sum_live_and_after_archive(
    accounting_pool: asyncpg.Pool[Any],
) -> None:
    root_id, run_id, child_id, _agent_id, _environment_id = await _seed_agent_workflow_agent_chain(
        accounting_pool
    )
    await _charge_known_chain(accounting_pool, root_id, run_id, child_id)

    # The ordinary resource reads expose the same accounting contract at
    # ``usage.own`` / ``usage.subtree`` on both node types.
    root_read = await sessions_service.get_session(accounting_pool, root_id, account_id=ACCOUNT)
    run_read = await workflows_service.get_run(accounting_pool, run_id, account_id=ACCOUNT)
    assert root_read.usage.own.cost_microusd == 100
    assert root_read.usage.subtree.cost_microusd == 600
    assert run_read.usage is not None
    assert run_read.usage.own.cost_microusd == 200
    assert run_read.usage.subtree.cost_microusd == 500

    async with accounting_pool.acquire() as conn:
        root = await accounting_queries.usage_for_node(
            conn,
            UsageNodeRef(kind="session", id=root_id),
            account_id=ACCOUNT,
            window_seconds=86_400,
        )
        run = await accounting_queries.usage_for_node(
            conn,
            UsageNodeRef(kind="run", id=run_id),
            account_id=ACCOUNT,
            window_seconds=86_400,
        )
        child = await accounting_queries.usage_for_node(
            conn,
            UsageNodeRef(kind="session", id=child_id),
            account_id=ACCOUNT,
            window_seconds=86_400,
        )
        assert root is not None and run is not None and child is not None
        assert [root.own.cost_microusd, run.own.cost_microusd, child.own.cost_microusd] == [
            100,
            200,
            300,
        ]
        assert root.subtree.cost_microusd == 600
        assert root.subtree.input_tokens == 60
        assert run.subtree.cost_microusd == 500
        assert root.subtree_rate is not None
        assert root.subtree_rate.cost_microusd_per_hour > 0

        # Finished work remains owned. A later child charge also updates the
        # already-finished ancestors live.
        await conn.execute("UPDATE sessions SET archived_at = now() WHERE id = $1", child_id)
        await conn.execute(
            "UPDATE wf_runs SET status = 'errored', archived_at = now() WHERE id = $1", run_id
        )
        await queries.increment_session_usage(
            conn,
            child_id,
            account_id=ACCOUNT,
            input_tokens=4,
            output_tokens=0,
            cost_microusd=40,
        )
        after_archive = await accounting_queries.usage_for_node(
            conn,
            UsageNodeRef(kind="session", id=root_id),
            account_id=ACCOUNT,
            window_seconds=86_400,
        )
        assert after_archive is not None
        assert after_archive.subtree.cost_microusd == 640
        assert after_archive.subtree.input_tokens == 64


async def test_descendant_mutation_moves_root_by_exact_amount_and_peer_invocation_moves_none(
    accounting_pool: asyncpg.Pool[Any],
) -> None:
    root_id, run_id, child_id, agent_id, environment_id = await _seed_agent_workflow_agent_chain(
        accounting_pool
    )
    await _charge_known_chain(accounting_pool, root_id, run_id, child_id)
    _agent2, _environment2, peer_caller = await seed_agent_env_session(
        accounting_pool, account_id=ACCOUNT, prefix="creation-peer"
    )

    async with accounting_pool.acquire() as conn:
        before = await accounting_queries.usage_for_nodes(
            conn,
            [
                UsageNodeRef(kind="session", id=root_id),
                UsageNodeRef(kind="session", id=peer_caller.id),
            ],
            account_id=ACCOUNT,
            window_seconds=86_400,
        )

        grandchild = await queries.insert_session(
            conn,
            account_id=ACCOUNT,
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=None,
            title="known mutation",
            metadata={},
            creator_session_id=child_id,
        )
        await queries.increment_session_usage(
            conn,
            grandchild.id,
            account_id=ACCOUNT,
            input_tokens=5,
            output_tokens=1,
            cost_microusd=55,
        )

        # This is the persisted shape of call_session against an existing peer.
        # It is an invocation edge only and must not rewrite the creation owner.
        await queries.append_event(
            conn,
            session_id=child_id,
            kind="lifecycle",
            data={
                "event": "request_opened",
                "request_id": "req_peer_invocation",
                "caller": {"kind": "session", "id": peer_caller.id, "awaited": True},
            },
            account_id=ACCOUNT,
        )

        after = await accounting_queries.usage_for_nodes(
            conn,
            [
                UsageNodeRef(kind="session", id=root_id),
                UsageNodeRef(kind="session", id=peer_caller.id),
            ],
            account_id=ACCOUNT,
            window_seconds=86_400,
        )
        assert after[("session", root_id)].subtree.cost_microusd == (
            before[("session", root_id)].subtree.cost_microusd + 55
        )
        assert (
            after[("session", peer_caller.id)].subtree
            == before[("session", peer_caller.id)].subtree
        )


async def test_cycle_is_bounded_and_each_node_counts_once(
    accounting_pool: asyncpg.Pool[Any],
) -> None:
    root_id, run_id, child_id, _agent_id, _environment_id = await _seed_agent_workflow_agent_chain(
        accounting_pool
    )
    await _charge_known_chain(accounting_pool, root_id, run_id, child_id)
    async with accounting_pool.acquire() as conn:
        # Simulate a malformed legacy re-entry cycle. Recursive UNION globally
        # deduplicates (kind,id), so the walk terminates without double counting.
        await conn.execute(
            "UPDATE sessions SET creator_session_id = $1 WHERE id = $2", child_id, root_id
        )
        usage = await accounting_queries.usage_for_node(
            conn,
            UsageNodeRef(kind="session", id=root_id),
            account_id=ACCOUNT,
            window_seconds=86_400,
        )
        assert usage is not None
        assert usage.subtree.cost_microusd == 600


async def test_ranked_view_fingers_known_hot_root_in_one_query(
    accounting_pool: asyncpg.Pool[Any],
) -> None:
    root_id, run_id, child_id, _agent_id, _environment_id = await _seed_agent_workflow_agent_chain(
        accounting_pool
    )
    await _charge_known_chain(accounting_pool, root_id, run_id, child_id)
    _agent2, _environment2, hot = await seed_agent_env_session(
        accounting_pool, account_id=ACCOUNT, prefix="creation-hot"
    )
    async with accounting_pool.acquire() as conn:
        await queries.increment_session_usage(
            conn,
            hot.id,
            account_id=ACCOUNT,
            input_tokens=1_000,
            output_tokens=100,
            cost_microusd=10_000,
        )
        coverage, total_rate, consumers = await accounting_queries.ranked_consumers(
            conn,
            account_id=ACCOUNT,
            window_seconds=86_400,
            metric="cost_microusd",
            limit=20,
        )
        assert coverage is not None
        assert total_rate > 0
        assert consumers[0].id == hot.id
        assert consumers[0].usage.subtree.cost_microusd == 10_000
        assert sum(item.share for item in consumers) == pytest.approx(1.0)
