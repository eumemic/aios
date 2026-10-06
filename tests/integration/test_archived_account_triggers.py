"""An archived account's triggers stop firing, and resume when it is unarchived.

Nothing fires on behalf of an archived account: the scheduler's claim and
next-event queries skip its triggers (session and operator alike), an ingest token
of its resolves to nothing, a run completing in it inserts no fires, and a fire
claimed just before the archive is skipped at execute.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.harness.trigger_runner import run_trigger_step
from aios.models.triggers import OperatorTriggerCreate, TriggerCreate
from aios.services import triggers as trig_service
from aios.workflows import run_tools
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

LIVE = "acc_arch_live"
GONE = "acc_arch_gone"


@pytest.fixture
async def arch_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            for account, parent in ((LIVE, None), (GONE, LIVE)):
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


async def _set_archived(pool: asyncpg.Pool[Any], account_id: str, archived: bool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE accounts SET archived_at = CASE WHEN $2 THEN now() END WHERE id = $1",
            account_id,
            archived,
        )


async def _workflow(pool: asyncpg.Pool[Any], account_id: str) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id=account_id,
            name=f"w-{secrets.token_hex(4)}",
            script="async def main(input):\n    return input\n",
        )
    return wf.id


async def _cron_triggers(pool: asyncpg.Pool[Any], account_id: str) -> tuple[str, str]:
    """A session cron trigger and an operator cron trigger in ``account_id``."""
    _, env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix=f"arch-{secrets.token_hex(3)}"
    )
    session_trigger = await trig_service.add_trigger(
        pool,
        session.id,
        TriggerCreate.model_validate(
            {
                "name": "tick",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "wake_owner", "content": "tick"},
            }
        ),
        account_id=account_id,
    )
    operator_trigger = await trig_service.add_operator_trigger(
        pool,
        OperatorTriggerCreate.model_validate(
            {
                "name": "weekly",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {
                    "kind": "workflow",
                    "workflow_id": await _workflow(pool, account_id),
                    "budget_usd": 1.0,
                },
                "environment_id": env.id,
            }
        ),
        account_id=account_id,
    )
    return session_trigger.id, operator_trigger.id


async def _claim(pool: asyncpg.Pool[Any]) -> set[str]:
    async with pool.acquire() as conn, conn.transaction():
        far_future = (await conn.fetchval("SELECT now()")) + timedelta(days=1)
        return {t.id for t in await queries.fetch_and_claim_due_triggers(conn, now_utc=far_future)}


async def test_an_archived_accounts_triggers_are_not_claimed_and_resume_when_unarchived(
    arch_runtime: asyncpg.Pool[Any],
) -> None:
    pool = arch_runtime
    live = set(await _cron_triggers(pool, LIVE))
    gone = set(await _cron_triggers(pool, GONE))
    await _set_archived(pool, GONE, True)

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE triggers SET next_fire = now() + interval '1 hour' WHERE account_id = $1",
            LIVE,
        )
        await conn.execute(
            "UPDATE triggers SET next_fire = now() - interval '1 hour' WHERE account_id = $1",
            GONE,
        )
        live_next = await conn.fetchval(
            "SELECT min(next_fire) FROM triggers WHERE account_id = $1", LIVE
        )
        # The archived account's overdue triggers don't pull the scheduler's wake in.
        assert await queries.fetch_next_trigger_event(conn) == live_next

    assert await _claim(pool) == live

    await _set_archived(pool, GONE, False)
    assert await _claim(pool) == gone


async def test_an_archived_accounts_ingest_token_resolves_to_nothing(
    arch_runtime: asyncpg.Pool[Any],
) -> None:
    pool = arch_runtime
    _, _, session = await seed_agent_env_session(pool, account_id=GONE, prefix="arch-ingest")
    created = await trig_service.add_trigger(
        pool,
        session.id,
        TriggerCreate.model_validate(
            {
                "name": "hook",
                "source": {"kind": "external_event"},
                "action": {"kind": "wake_owner", "content": "ping"},
            }
        ),
        account_id=GONE,
    )
    assert created.ingest_token is not None
    token_hash = hashlib.sha256(created.ingest_token.encode("utf-8")).hexdigest()
    await _set_archived(pool, GONE, True)

    async with pool.acquire() as conn:
        assert (
            await queries.resolve_external_event_trigger(conn, ingest_token_hash=token_hash) is None
        )


async def test_a_run_completing_in_an_archived_account_fires_nothing(
    arch_runtime: asyncpg.Pool[Any],
) -> None:
    pool = arch_runtime
    _, _, session = await seed_agent_env_session(pool, account_id=GONE, prefix="arch-rc")
    wf_id = await _workflow(pool, GONE)
    await trig_service.add_trigger(
        pool,
        session.id,
        TriggerCreate.model_validate(
            {
                "name": "on-done",
                "source": {"kind": "run_completion", "workflow_id": wf_id},
                "action": {"kind": "wake_owner", "content": "done"},
            }
        ),
        account_id=GONE,
    )
    await _set_archived(pool, GONE, True)

    async with pool.acquire() as conn:
        fires = await queries.insert_run_completion_fires(
            conn,
            account_id=GONE,
            workflow_id=wf_id,
            run_id="wfr_arch",
            status="completed",
            visibility="account",
            launcher_session_id=None,
        )
    assert fires == []


async def test_a_fire_claimed_before_the_archive_is_skipped(
    arch_runtime: asyncpg.Pool[Any],
) -> None:
    pool = arch_runtime
    _, operator_id = await _cron_triggers(pool, GONE)
    assert operator_id in await _claim(pool)
    await _set_archived(pool, GONE, True)

    await run_trigger_step(operator_id)

    async with pool.acquire() as conn:
        runs = await conn.fetchval(
            "SELECT count(*) FROM wf_runs WHERE trigger_id = $1", operator_id
        )
        status = await conn.fetchval(
            "SELECT last_fire_status FROM triggers WHERE id = $1", operator_id
        )
    assert runs == 0
    assert status == "skipped"
