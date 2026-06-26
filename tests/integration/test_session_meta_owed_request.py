"""Integration tests for ``read_session_meta_batched``'s open-request liveness.

The trace normalizer's archived-root green-washing check keys off
``read_session_meta_batched(...)[sid]["open_request_ids"]`` ("does this archived
root still owe a request?") and the derived ``owed_request_response``. That
"owes a request" set must mean "owes a *response*" — exactly
:func:`aios.db.queries.sessions.get_open_request_ids`'s awaited-only open set —
so a ``Tell(NewSession)`` fire-and-forget spawn (an **unawaited**
``request_opened`` edge that owes no response) does NOT count as open. Before
#1536 this reader omitted the ``awaited`` filter, so an archived root whose only
edge was an unawaited ``Tell`` spawn was wrongly counted as owing a request and
the ``_resolve_owed_response`` path resolved that phantom obligation to
``child_gone`` — mislabeling a clean archived root as errored.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import trace as trace_q
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_session_meta_owed"


@pytest.fixture
async def pool_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, agent_id, environment_id)`` for a fresh tenant."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'session-meta-owed-test')",
                _ACCOUNT,
            )
        agent, env, _session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="session-meta-owed"
        )
        yield pool, agent.id, env.id
    finally:
        await pool.close()


async def _insert_session(pool: asyncpg.Pool[Any], *, agent_id: str, environment_id: str) -> str:
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            account_id=_ACCOUNT,
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=1,
            title=None,
            metadata={},
        )
    return session.id


async def _open_edge(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    request_id: str,
    caller: dict[str, Any],
    awaited: bool,
) -> None:
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            caller=caller,
            depth=0,
            environment_id="env_x",
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            awaited=awaited,
        )


async def _archive(pool: asyncpg.Pool[Any], *, session_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET archived_at = now() WHERE id = $1 AND account_id = $2",
            session_id,
            _ACCOUNT,
        )


async def test_archived_unawaited_tell_owes_nothing(
    pool_env: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """An archived session whose only ``request_opened`` edge is an **unawaited**
    ``Tell(NewSession)`` spawn (``awaited=false``, no ``request_response``) owes
    nothing: ``open_request_ids == []`` and ``owed_request_response is None``.

    On ``master`` this fails — the unawaited edge was wrongly counted as open and
    the archived-root resolution turned it into a phantom ``child_gone`` (green-
    washing a clean archived root into ``errored``). After #1536's shared
    ``awaited_only=True`` fragment it passes.
    """
    pool, agent_id, env_id = pool_env
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id)
    await _open_edge(
        pool,
        session_id=sid,
        request_id="req_tell",
        caller={"kind": "run", "id": "wfr_tell"},
        awaited=False,
    )
    await _archive(pool, session_id=sid)
    async with pool.acquire() as conn:
        meta = await trace_q.read_session_meta_batched(conn, [sid], account_id=_ACCOUNT)
    assert meta[sid]["open_request_ids"] == []
    assert meta[sid]["owed_request_response"] is None


async def test_archived_awaited_ask_still_owes(
    pool_env: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """The behavior-preserving control: an archived session with an **awaited**
    (``Ask``-arm) open edge and no response still owes — ``open_request_ids`` is
    non-empty and ``owed_request_response`` resolves to the ``child_gone``
    terminal outcome (an archived session can never answer)."""
    pool, agent_id, env_id = pool_env
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id)
    await _open_edge(
        pool,
        session_id=sid,
        request_id="req_ask",
        caller={"kind": "run", "id": "wfr_ask"},
        awaited=True,
    )
    await _archive(pool, session_id=sid)
    async with pool.acquire() as conn:
        meta = await trace_q.read_session_meta_batched(conn, [sid], account_id=_ACCOUNT)
    assert meta[sid]["open_request_ids"] == ["req_ask"]
    owed = meta[sid]["owed_request_response"]
    assert owed is not None and owed["is_error"] is True
    assert owed["error"] == {"kind": "child_gone"}
