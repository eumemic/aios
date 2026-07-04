"""Integration tests for the API caller's request-writer — ``service.invoke`` (#1128).

DB-backed (testcontainer Postgres). Exercises the kind-agnostic request-writer end
to end against real rows:

* ``target_kind=agent`` creates a **session** servicer, writes the request edge
  (#1123) with ``caller={kind:"api", ...}``, and the returned ``request_id``
  correlates a subsequent ``await_session`` to the agent's response.
* ``target_kind=workflow`` creates a **run** servicer (``servicer_kind="run"``).
* ``target_kind=session`` invokes an existing same-account session.
* A cross-tenant ``target`` 404s before any edge is written.
* An ``environment_id`` not owned by the caller's account is refused.
* A ``target_kind``/``target`` mismatch is refused before any edge is written.

The session-wake / run-wake defers are patched out — these tests cover the
edge-write + servicer resolution, not the worker stepping.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError, ValidationError
from aios.harness import runtime
from aios.models.sessions import Ok
from aios.services import sessions as service
from aios.services import tasks as tasks_service
from aios.services import workflows as wf_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ROOT = "acc_tasks_root"
_ACCOUNT = "acc_tasks"
_OTHER = "acc_tasks_other"


@pytest.fixture
async def pool_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str, str]]:
    """Yield ``(pool, account_id, agent_id, environment_id, session_id)`` for a fresh tenant.

    Patches out both the session-wake and run-wake defers so the request-write +
    servicer-resolution paths run without a live procrastinate worker.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            # A single active root is permitted (accounts_one_active_root); the two
            # test tenants are sibling children of that root.
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, NULL, TRUE, 'tasks-root')",
                _ROOT,
            )
            for acc in (_ACCOUNT, _OTHER):
                await conn.execute(
                    "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                    "display_name) VALUES ($1, $2, FALSE, $3)",
                    acc,
                    _ROOT,
                    acc,
                )
        agent, env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="tasks"
        )
        with (
            mock.patch("aios.services.sessions.defer_wake", new=AsyncMock()),
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
        ):
            yield pool, _ACCOUNT, agent.id, env.id, session.id
    finally:
        runtime.pool = prev
        await pool.close()


async def _seed_workflow(pool: asyncpg.Pool[Any], *, account_id: str) -> str:
    wf = await wf_service.create_workflow(
        pool,
        account_id=account_id,
        name="tasks-wf",
        script="async def main(input):\n    return None\n",
        description=None,
        tools=[],
    )
    return wf.id


# ─── target_kind=agent → session servicer ────────────────────────────────────


async def test_agent_target_creates_session_and_writes_edge(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, agent_id, env_id, _session_id = pool_env

    handle = await service.invoke(
        pool,
        account_id=account_id,
        target_kind="agent",
        target=agent_id,
        input={"q": "hello"},
        environment_id=env_id,
    )

    assert handle.servicer_kind == "session"
    assert handle.servicer_id.startswith("sess_")
    assert handle.request_id.startswith("req_")

    # The session exists, is bound to the right agent/env, and owns the open request.
    sess = await service.get_session(pool, handle.servicer_id, account_id=account_id)
    assert sess.agent_id == agent_id
    assert sess.environment_id == env_id
    async with pool.acquire() as conn:
        open_ids = await queries.get_open_request_ids(
            conn, handle.servicer_id, account_id=account_id
        )
    assert open_ids == [handle.request_id]


async def test_agent_request_correlates_await_to_response(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, agent_id, env_id, _session_id = pool_env

    handle = await service.invoke(
        pool,
        account_id=account_id,
        target_kind="agent",
        target=agent_id,
        input="solve it",
        environment_id=env_id,
    )

    # Simulate the agent answering the request (what return() writes).
    async with pool.acquire() as conn:
        await queries.write_response_if_absent(
            conn,
            handle.servicer_id,
            account_id=account_id,
            request_id=handle.request_id,
            outcome=Ok(result={"answer": 42}),
        )

    resp = await tasks_service.await_task(
        pool,
        migrated_db_url,
        servicer_kind="session",
        servicer_id=handle.servicer_id,
        request_id=handle.request_id,
        account_id=account_id,
        timeout_seconds=5,
    )
    assert resp.outcome == "ok"
    assert resp.result == {"answer": 42}


async def test_agent_output_schema_rides_the_edge(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, agent_id, env_id, _session_id = pool_env
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}

    handle = await service.invoke(
        pool,
        account_id=account_id,
        target_kind="agent",
        target=agent_id,
        input="x",
        output_schema=schema,
        environment_id=env_id,
    )

    # The schema rides metadata.request.output_schema on the injected request message.
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'message' ORDER BY seq",
            handle.servicer_id,
        )

    metas = [r["data"].get("metadata", {}).get("request") for r in rows]
    matching = [m for m in metas if m and m.get("request_id") == handle.request_id]
    assert matching and matching[0]["output_schema"] == schema
    # Channel-less: the request never carries a connector channel.
    assert "channel" not in (matching[0] or {})


# ─── target_kind=session → existing session servicer ─────────────────────────


async def test_session_target_invokes_existing_session(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, _env_id, session_id = pool_env

    handle = await service.invoke(
        pool,
        account_id=account_id,
        target_kind="session",
        target=session_id,
        input="ping",
    )
    assert handle.servicer_kind == "session"
    assert handle.servicer_id == session_id
    async with pool.acquire() as conn:
        open_ids = await queries.get_open_request_ids(conn, session_id, account_id=account_id)
    assert handle.request_id in open_ids


async def test_session_target_rejects_environment_id(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id, session_id = pool_env
    with pytest.raises(ValidationError):
        await service.invoke(
            pool,
            account_id=account_id,
            target_kind="session",
            target=session_id,
            input="ping",
            environment_id=env_id,
        )


# ─── target_kind=workflow → run servicer ─────────────────────────────────────


async def test_workflow_target_creates_run(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id, _session_id = pool_env
    wf_id = await _seed_workflow(pool, account_id=account_id)

    handle = await service.invoke(
        pool,
        account_id=account_id,
        target_kind="workflow",
        target=wf_id,
        input={"n": 3},
        environment_id=env_id,
    )
    assert handle.servicer_kind == "run"
    assert handle.servicer_id.startswith("wfr_")
    assert handle.request_id.startswith("req_")
    # The run row exists and is account-scoped.
    run = await wf_service.get_run(pool, handle.servicer_id, account_id=account_id)
    assert run.workflow_id == wf_id


# ─── auth + ownership + validation ───────────────────────────────────────────


async def test_cross_tenant_agent_target_404s_before_edge(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, _account_id, agent_id, env_id, _session_id = pool_env
    # The agent belongs to _ACCOUNT; the caller is _OTHER → 404 before any write.
    with pytest.raises(NotFoundError):
        await service.invoke(
            pool,
            account_id=_OTHER,
            target_kind="agent",
            target=agent_id,
            input="x",
            environment_id=env_id,
        )
    # No session was created for _OTHER.
    sessions = await service.list_sessions(pool, account_id=_OTHER)
    assert sessions == []


async def test_cross_tenant_session_target_404s(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, _account_id, _agent_id, _env_id, session_id = pool_env
    with pytest.raises(NotFoundError):
        await service.invoke(
            pool,
            account_id=_OTHER,
            target_kind="session",
            target=session_id,
            input="x",
        )


async def test_unowned_environment_refused(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, agent_id, _env_id, _session_id = pool_env
    # Seed an env owned by the OTHER account.
    from aios.services import environments as env_service

    other_env = await env_service.create_environment(
        pool, account_id=_OTHER, name="tasks-other-env"
    )
    with pytest.raises(NotFoundError):
        await service.invoke(
            pool,
            account_id=account_id,
            target_kind="agent",
            target=agent_id,
            input="x",
            environment_id=other_env.id,
        )


async def test_target_kind_target_mismatch_refused(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id, _session_id = pool_env
    wf_id = await _seed_workflow(pool, account_id=account_id)
    # A workflow_id under target_kind=agent must not resolve to an agent.
    with pytest.raises(NotFoundError):
        await service.invoke(
            pool,
            account_id=account_id,
            target_kind="agent",
            target=wf_id,
            input="x",
            environment_id=env_id,
        )


async def test_unknown_target_kind_refused(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, _env_id, _session_id = pool_env
    with pytest.raises(ValidationError):
        await service.invoke(
            pool,
            account_id=account_id,
            target_kind="robot",
            target="x",
            input="y",
        )
