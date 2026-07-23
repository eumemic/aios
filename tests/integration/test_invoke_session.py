"""Integration tests for the **session caller surface** (#1127).

DB-backed (testcontainer Postgres). Covers the session-caller half that the API
caller (#1128, ``test_tasks.py``) does not:

* ``service.invoke(target_kind=session, caller={kind:"session", ...})`` writes the
  trusted ``request_opened`` edge into an EXISTING same-account session with a
  ``caller.kind == "session"`` provenance — and the injected request is
  **channel-less** (no ``orig_channel``), so it never surfaces to a connector.
* A NON-child session that owns an open request can **answer** via
  ``respond_to_request`` — the return/error gate re-key (#1123/#1131 fold): the
  response edge is written and ``derive_response`` resolves it.
* A cross-account ``session_id`` **404s before any edge is written**.
* the return/error injection gate (``bool(get_open_obligations(...))`` since #1413
  superseded ``session_owns_open_request``) flips True exactly when a session owes
  a request.
* The totality backstop auto-errors a non-child session that idles while owing,
  routing the wake to its session caller (no run involved).
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
from aios.errors import ForbiddenError, NotFoundError
from aios.harness import runtime
from aios.models.agents import ToolSpec
from aios.models.sessions import Ok
from aios.models.tasks import TaskHandle
from aios.services import agents as agents_service
from aios.services import sessions as service
from aios.services import vaults as vaults_service
from aios.tools import workflow_completion
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ROOT = "acc_invsess_root"
_ACCOUNT = "acc_invsess"
_OTHER = "acc_invsess_other"


@pytest.fixture
async def pool_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str, str]]:
    """Yield ``(pool, account_id, agent_id, environment_id, session_id)``.

    Patches the wake defers so the request-write path runs without a live worker.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, NULL, TRUE, 'invsess-root')",
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
            pool, account_id=_ACCOUNT, prefix="invsess"
        )
        with (
            mock.patch("aios.services.sessions.defer_wake", new=AsyncMock()),
            mock.patch("aios.jobs.app.defer_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
        ):
            yield pool, _ACCOUNT, agent.id, env.id, session.id
    finally:
        runtime.pool = prev
        await pool.close()


async def _invoke_session(
    pool: asyncpg.Pool[Any], *, account_id: str, caller_session_id: str, target: str, **kw: Any
) -> Any:
    return await service.invoke(
        pool,
        account_id=account_id,
        target_kind="session",
        target=target,
        input=kw.get("input", "do it"),
        output_schema=kw.get("output_schema"),
        caller={"kind": "session", "id": caller_session_id},
    )


async def test_invoke_existing_session_writes_caller_session_edge(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, _env_id, session_id = pool_env
    # The caller invokes itself's sibling: seed a second session as the target.
    _, _, target = await seed_agent_env_session(pool, account_id=account_id, prefix="invsess_t")

    handle = await _invoke_session(
        pool, account_id=account_id, caller_session_id=session_id, target=target.id
    )

    assert handle.servicer_kind == "session"
    assert handle.servicer_id == target.id
    async with pool.acquire() as conn:
        open_ids = await queries.get_open_request_ids(conn, target.id, account_id=account_id)
        caller = await queries.get_request_caller(conn, target.id, request_id=handle.request_id)
    assert open_ids == [handle.request_id]
    assert caller == {"kind": "session", "id": session_id}


async def test_injected_request_is_channel_less(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, _env_id, session_id = pool_env
    _, _, target = await seed_agent_env_session(pool, account_id=account_id, prefix="invsess_cl")

    await _invoke_session(
        pool, account_id=account_id, caller_session_id=session_id, target=target.id
    )

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT orig_channel, data FROM events WHERE session_id = $1 AND kind = 'message' "
            "AND role = 'user' ORDER BY seq",
            target.id,
        )
    # The request message carries no connector channel — invisible to any human chat.
    assert rows, "expected the injected request message"
    for r in rows:
        assert r["orig_channel"] is None

    metas = [r["data"].get("metadata", {}) for r in rows]
    assert all("channel" not in m for m in metas)


async def test_cross_account_target_404s_before_edge(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, _env_id, session_id = pool_env
    # A session owned by a DIFFERENT account.
    _, _, foreign = await seed_agent_env_session(pool, account_id=_OTHER, prefix="invsess_foreign")

    with pytest.raises(NotFoundError):
        await _invoke_session(
            pool, account_id=account_id, caller_session_id=session_id, target=foreign.id
        )

    # No edge was written into the foreign session.
    async with pool.acquire() as conn:
        open_ids = await queries.get_open_request_ids(conn, foreign.id, account_id=_OTHER)
    assert open_ids == []


async def test_non_child_session_answers_via_respond_to_request(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """A non-child session that owns an open request can answer — gate re-key (#1127)."""
    pool, account_id, _agent_id, _env_id, session_id = pool_env
    _, _, target = await seed_agent_env_session(pool, account_id=account_id, prefix="invsess_ans")

    handle = await _invoke_session(
        pool, account_id=account_id, caller_session_id=session_id, target=target.id
    )

    status = await workflow_completion.respond_to_request(
        pool,
        target.id,
        request_id=handle.request_id,
        outcome=Ok(result={"answer": 7}),
    )
    assert status == "responded"

    async with pool.acquire() as conn:
        resolved = await queries.derive_response(
            conn, target.id, account_id=account_id, request_id=handle.request_id
        )
    assert resolved == Ok(result={"answer": 7})
    # Single-shot: the request is now answered (no longer open).
    async with pool.acquire() as conn:
        open_ids = await queries.get_open_request_ids(conn, target.id, account_id=account_id)
    assert open_ids == []


async def test_session_owns_open_request_gate(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """The return/error injection gate flips on exactly while a request is open.

    #1413 deleted ``session_owns_open_request``; the gate is now ``bool(obligations)``
    where ``obligations = get_open_obligations(...)`` (the same awaited anti-join,
    correctness-equivalent). Assert the gate semantics on the open-obligations set.
    """
    pool, account_id, _agent_id, _env_id, session_id = pool_env
    _, _, target = await seed_agent_env_session(pool, account_id=account_id, prefix="invsess_gate")

    async def _owes() -> bool:
        async with pool.acquire() as conn:
            return bool(await queries.get_open_obligations(conn, target.id, account_id=account_id))

    assert not await _owes()

    handle = await _invoke_session(
        pool, account_id=account_id, caller_session_id=session_id, target=target.id
    )
    assert await _owes()

    await workflow_completion.respond_to_request(
        pool, target.id, request_id=handle.request_id, outcome=Ok(result="ok")
    )
    assert not await _owes()


async def test_duplicate_answer_is_idempotent(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, _env_id, session_id = pool_env
    _, _, target = await seed_agent_env_session(pool, account_id=account_id, prefix="invsess_dup")
    handle = await _invoke_session(
        pool, account_id=account_id, caller_session_id=session_id, target=target.id
    )

    first = await workflow_completion.respond_to_request(
        pool, target.id, request_id=handle.request_id, outcome=Ok(result=1)
    )
    second = await workflow_completion.respond_to_request(
        pool, target.id, request_id=handle.request_id, outcome=Ok(result=2)
    )
    assert first == "responded"
    assert second == "duplicate"  # first-writer-wins


async def _invoke_agent_child(
    pool: asyncpg.Pool[Any],
    account_id: str,
    parent_id: str,
    agent_id: str,
    env_id: str,
    **kw: Any,
) -> TaskHandle:
    return await service.invoke(
        pool,
        account_id=account_id,
        target_kind="agent",
        target=agent_id,
        environment_id=env_id,
        input="go",
        launcher_session_id=parent_id,
        caller={"kind": "session", "id": parent_id},
        **kw,
    )


async def test_call_agent_vault_attenuation_and_suppression(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, agent_id, env_id, parent_id = pool_env
    first = await vaults_service.create_vault(
        pool, account_id=account_id, display_name="first", metadata={}
    )
    second = await vaults_service.create_vault(
        pool, account_id=account_id, display_name="second", metadata={}
    )
    outside = await vaults_service.create_vault(
        pool, account_id=account_id, display_name="outside", metadata={}
    )
    await service.update_session(
        pool,
        parent_id,
        account_id=account_id,
        vault_ids=[first.id, second.id],
        outbound_suppression="on",
    )

    inherited = await _invoke_agent_child(pool, account_id, parent_id, agent_id, env_id)
    inherited_session = await service.get_session(
        pool, inherited.servicer_id, account_id=account_id
    )
    assert set(inherited_session.vault_ids) == {first.id, second.id}
    assert inherited_session.outbound_suppression == "on"

    empty = await _invoke_agent_child(
        pool, account_id, parent_id, agent_id, env_id, vault_ids=[], outbound_suppression="off"
    )
    empty_session = await service.get_session(pool, empty.servicer_id, account_id=account_id)
    assert empty_session.vault_ids == []
    assert empty_session.outbound_suppression == "on"

    import pytest as _pytest

    with _pytest.raises(ForbiddenError, match="vault"):
        await _invoke_agent_child(
            pool, account_id, parent_id, agent_id, env_id, vault_ids=[outside.id]
        )


async def test_call_agent_frozen_surface_survives_reload_and_composes(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    pool, account_id, _agent_id, env_id, parent_id = pool_env
    parent_agent, _, _ = await seed_agent_env_session(
        pool, account_id=account_id, prefix="clamped_parent", tools=[ToolSpec(type="bash")]
    )
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET agent_id=$1, agent_version=$2 WHERE id=$3",
            parent_agent.id,
            parent_agent.version,
            parent_id,
        )
    broad, _, _ = await seed_agent_env_session(
        pool,
        account_id=account_id,
        prefix="broad_child",
        tools=[ToolSpec(type="bash"), ToolSpec(type="read")],
    )

    child_handle = await _invoke_agent_child(pool, account_id, parent_id, broad.id, env_id)
    child = await service.get_session_basic(pool, child_handle.servicer_id, account_id=account_id)
    assert child.surface_frozen is True
    loaded = await agents_service.load_for_session(pool, child, account_id=account_id)
    assert {tool.type for tool in loaded.tools} == {"bash"}

    grandchild_handle = await _invoke_agent_child(pool, account_id, child.id, broad.id, env_id)
    grandchild = await service.get_session_basic(
        pool, grandchild_handle.servicer_id, account_id=account_id
    )
    grandchild_agent = await agents_service.load_for_session(
        pool, grandchild, account_id=account_id
    )
    assert {tool.type for tool in grandchild_agent.tools} == {"bash"}
