"""Operator-owned triggers (#2473) against a real Postgres.

Drives the service layer, the scheduler's claim and next-event queries, and
``run_trigger_step`` directly with every procrastinate defer patched out (the
``test_trigger_event_fires.py`` surface). Covers the acceptance list:

- an operator cron trigger fires on schedule, and its run is an operator run;
  a one-shot operator trigger fires once and deletes itself;
- an agent can't see, change or remove an operator trigger through the
  session-keyed trigger surface;
- two operator triggers with one name in an account collide;
- the breaker auto-disables an operator trigger without touching any session.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError, NotFoundError
from aios.harness import runtime
from aios.harness.trigger_runner import MAX_CONSECUTIVE_FAILURES, run_trigger_step
from aios.models.agents import ToolSpec
from aios.models.triggers import (
    OperatorTriggerCreate,
    OperatorTriggerUpdate,
    TriggerCreate,
    TriggerUpdate,
)
from aios.services import triggers as trig_service
from aios.services import workflows as wf_service
from aios.workflows import run_tools
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

ACC = "acc_optrig"
OTHER = "acc_optrig_other"


@pytest.fixture
async def op_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            for account, parent in ((ACC, None), (OTHER, ACC)):
                await conn.execute(
                    "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                    "display_name) VALUES ($1, $2, TRUE, $1)",
                    account,
                    parent,
                )
        run_tools._INFLIGHT.clear()
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.sessions.defer_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _workflow(pool: asyncpg.Pool[Any], account_id: str = ACC) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id=account_id,
            name=f"w-{secrets.token_hex(4)}",
            script="async def main(input):\n    return input\n",
        )
    return wf.id


async def _scaffold(pool: asyncpg.Pool[Any], account_id: str = ACC) -> tuple[str, str, str]:
    """``(environment id, a session id, a workflow id)`` in ``account_id``."""
    _, env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix=f"op-{secrets.token_hex(3)}"
    )
    return env.id, session.id, await _workflow(pool, account_id)


def _spec(
    name: str,
    env_id: str,
    workflow_id: str,
    *,
    source: dict[str, Any] | None = None,
    vault_ids: list[str] | None = None,
) -> OperatorTriggerCreate:
    return OperatorTriggerCreate.model_validate(
        {
            "name": name,
            "source": source or {"kind": "cron", "schedule": "*/5 * * * *"},
            "action": {
                "kind": "workflow",
                "workflow_id": workflow_id,
                "input_template": {"week": 1},
                "vault_ids": vault_ids or [],
                "budget_usd": 2.5,
            },
            "environment_id": env_id,
        }
    )


async def _launched_runs(pool: asyncpg.Pool[Any], trigger_id: str) -> list[asyncpg.Record]:
    async with pool.acquire() as conn:
        return list(
            await conn.fetch(
                "SELECT * FROM wf_runs WHERE trigger_id = $1 ORDER BY created_at", trigger_id
            )
        )


async def test_an_operator_cron_trigger_fires_on_schedule_and_launches_an_operator_run(
    op_runtime: asyncpg.Pool[Any],
) -> None:
    pool = op_runtime
    env_id, _, wf_id = await _scaffold(pool)
    echo = await trig_service.add_operator_trigger(
        pool, _spec("weekly", env_id, wf_id), account_id=ACC
    )
    assert echo.environment_id == env_id and echo.next_fire is not None

    async with pool.acquire() as conn:
        assert await queries.fetch_next_trigger_event(conn) == echo.next_fire
    async with pool.acquire() as conn, conn.transaction():
        claimed = await queries.fetch_and_claim_due_triggers(
            conn, now_utc=echo.next_fire + timedelta(seconds=1)
        )
    [row] = [c for c in claimed if c.id == echo.id]
    assert row.owner == queries.OperatorOwner()

    await run_trigger_step(echo.id)

    [run] = await _launched_runs(pool, echo.id)
    assert run["principal"] == "operator"
    assert run["launcher_session_id"] is None
    assert run["parent_run_id"] is None
    assert run["environment_id"] == env_id
    assert run["budget_total_microusd"] == 2_500_000
    [fire] = await trig_service.list_operator_trigger_runs(pool, "weekly", account_id=ACC)
    assert fire.status == "ok" and fire.result_id == run["id"]
    async with pool.acquire() as conn:
        owner = await conn.fetchval(
            "SELECT owner_session_id FROM trigger_runs WHERE id = $1", fire.id
        )
    assert owner is None
    current = await trig_service.get_operator_trigger(pool, "weekly", account_id=ACC)
    assert current.last_fire_status == "ok"


async def test_a_one_shot_operator_trigger_fires_once_and_deletes_itself(
    op_runtime: asyncpg.Pool[Any],
) -> None:
    pool = op_runtime
    env_id, _, wf_id = await _scaffold(pool)
    fire_at = datetime.now(UTC) + timedelta(minutes=5)
    echo = await trig_service.add_operator_trigger(
        pool,
        _spec("once", env_id, wf_id, source={"kind": "one_shot", "fire_at": fire_at.isoformat()}),
        account_id=ACC,
    )
    async with pool.acquire() as conn, conn.transaction():
        claimed = await queries.fetch_and_claim_due_triggers(
            conn, now_utc=fire_at + timedelta(seconds=1)
        )
    assert [c.id for c in claimed if c.id == echo.id] == [echo.id]

    await run_trigger_step(echo.id)

    [run] = await _launched_runs(pool, echo.id)
    assert run["principal"] == "operator"
    with pytest.raises(NotFoundError):
        await trig_service.get_operator_trigger(pool, "once", account_id=ACC)
    [fire] = await trig_service.list_operator_trigger_runs(pool, "once", account_id=ACC)
    assert fire.trigger_context == "one_shot" and fire.status == "ok"


async def test_an_agent_cannot_see_change_or_remove_an_operator_trigger(
    op_runtime: asyncpg.Pool[Any],
) -> None:
    """The agent trigger tools and session routes go through these session-keyed
    service calls, which never match an operator row."""
    pool = op_runtime
    env_id, session_id, wf_id = await _scaffold(pool)
    await trig_service.add_operator_trigger(pool, _spec("weekly", env_id, wf_id), account_id=ACC)

    with pytest.raises(NotFoundError):
        await trig_service.update_trigger(
            pool, session_id, "weekly", TriggerUpdate(enabled=False), account_id=ACC
        )
    with pytest.raises(NotFoundError):
        await trig_service.remove_trigger(pool, session_id, "weekly", account_id=ACC)
    assert await trig_service.list_triggers(pool, session_id, account_id=ACC) == []
    assert await trig_service.list_account_triggers(pool, account_id=ACC) == []
    assert await trig_service.list_trigger_runs(pool, session_id, "weekly", account_id=ACC) == []

    current = await trig_service.get_operator_trigger(pool, "weekly", account_id=ACC)
    assert current.enabled


async def test_operator_trigger_names_are_unique_per_account(
    op_runtime: asyncpg.Pool[Any],
) -> None:
    pool = op_runtime
    env_id, session_id, wf_id = await _scaffold(pool)
    await trig_service.add_operator_trigger(pool, _spec("weekly", env_id, wf_id), account_id=ACC)

    with pytest.raises(ConflictError):
        await trig_service.add_operator_trigger(
            pool, _spec("weekly", env_id, wf_id), account_id=ACC
        )
    other_env, _, other_wf = await _scaffold(pool, OTHER)
    await trig_service.add_operator_trigger(
        pool, _spec("weekly", other_env, other_wf), account_id=OTHER
    )
    # A session trigger in the same account may use the name too.
    await trig_service.add_trigger(
        pool,
        session_id,
        TriggerCreate.model_validate(
            {
                "name": "weekly",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "wake_owner", "content": "go"},
            }
        ),
        account_id=ACC,
    )


async def test_an_operator_trigger_must_name_an_environment_in_its_account(
    op_runtime: asyncpg.Pool[Any],
) -> None:
    pool = op_runtime
    _, _, wf_id = await _scaffold(pool)
    other_env, _, _ = await _scaffold(pool, OTHER)
    with pytest.raises(NotFoundError):
        await trig_service.add_operator_trigger(
            pool, _spec("weekly", other_env, wf_id), account_id=ACC
        )


async def test_the_breaker_disables_an_operator_trigger_without_a_session(
    op_runtime: asyncpg.Pool[Any],
) -> None:
    """Every fire fails (a vault that doesn't exist); the fifth disables the trigger.
    No session hears about it, and nothing crashes for want of one."""
    pool = op_runtime
    env_id, _, wf_id = await _scaffold(pool)
    echo = await trig_service.add_operator_trigger(
        pool, _spec("broken", env_id, wf_id, vault_ids=["vlt_missing"]), account_id=ACC
    )
    async with pool.acquire() as conn:
        events_before = await conn.fetchval("SELECT count(*) FROM events")

    for _ in range(MAX_CONSECUTIVE_FAILURES):
        await run_trigger_step(echo.id)

    current = await trig_service.get_operator_trigger(pool, "broken", account_id=ACC)
    assert not current.enabled
    assert current.consecutive_failures == MAX_CONSECUTIVE_FAILURES
    assert current.last_fire_status == "error"
    fires = await trig_service.list_operator_trigger_runs(pool, "broken", account_id=ACC)
    assert [f.status for f in fires] == ["error"] * MAX_CONSECUTIVE_FAILURES
    assert await _launched_runs(pool, echo.id) == []
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM events") == events_before


async def test_re_enabling_an_operator_trigger_rearms_it_and_resets_failures(
    op_runtime: asyncpg.Pool[Any],
) -> None:
    pool = op_runtime
    env_id, _, wf_id = await _scaffold(pool)
    await trig_service.add_operator_trigger(pool, _spec("weekly", env_id, wf_id), account_id=ACC)

    off = await trig_service.update_operator_trigger(
        pool, "weekly", OperatorTriggerUpdate(enabled=False), account_id=ACC
    )
    assert not off.enabled and off.next_fire is None
    on = await trig_service.update_operator_trigger(
        pool, "weekly", OperatorTriggerUpdate(enabled=True), account_id=ACC
    )
    assert on.enabled and on.next_fire is not None and on.consecutive_failures == 0

    await trig_service.remove_operator_trigger(pool, "weekly", account_id=ACC)
    assert await trig_service.list_operator_triggers(pool, account_id=ACC) == []


async def _replay_workflow(pool: asyncpg.Pool[Any]) -> str:
    """An operator-authored workflow declaring both replay tools (#2475)."""
    wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name=f"eval-{secrets.token_hex(4)}",
        script="async def main(input):\n    return input\n",
        tools=[ToolSpec(type="sample_requests"), ToolSpec(type="get_request")],
    )
    return wf.id


async def test_an_operator_cron_trigger_fires_a_replay_workflow_as_a_private_operator_run(
    op_runtime: asyncpg.Pool[Any],
) -> None:
    """#2475: an operator trigger launches under the operator, so the run keeps both
    replay tools, and the 0191 visibility arm hides it from every agent."""
    pool = op_runtime
    env_id, _, _ = await _scaffold(pool)
    wf_id = await _replay_workflow(pool)
    echo = await trig_service.add_operator_trigger(
        pool, _spec("eval-weekly", env_id, wf_id), account_id=ACC
    )
    assert echo.next_fire is not None
    async with pool.acquire() as conn, conn.transaction():
        await queries.fetch_and_claim_due_triggers(
            conn, now_utc=echo.next_fire + timedelta(seconds=1)
        )

    await run_trigger_step(echo.id)

    [row] = await _launched_runs(pool, echo.id)
    assert row["principal"] == "operator"
    assert row["visibility"] == "session"
    assert row["launcher_session_id"] is None
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, row["id"])
    assert run is not None
    assert {t.type for t in run.tools} == {"sample_requests", "get_request"}


async def test_a_session_trigger_firing_a_replay_workflow_drops_the_replay_tools(
    op_runtime: asyncpg.Pool[Any],
) -> None:
    """#2475: a session-owned trigger launches under its session, whose agent can't
    hold a replay tool, so the run's surface is clamped to none of them."""
    pool = op_runtime
    _, session_id, _ = await _scaffold(pool)
    wf_id = await _replay_workflow(pool)
    echo = await trig_service.add_trigger(
        pool,
        session_id,
        TriggerCreate.model_validate(
            {
                "name": "session-eval",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "workflow", "workflow_id": wf_id},
            }
        ),
        account_id=ACC,
    )
    await run_trigger_step(echo.id)

    [row] = await _launched_runs(pool, echo.id)
    assert row["principal"] == "session"
    assert row["launcher_session_id"] == session_id
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, row["id"])
    assert run is not None
    assert {t.type for t in run.tools} & {"sample_requests", "get_request"} == set()
