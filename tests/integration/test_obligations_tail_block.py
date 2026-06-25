"""Integration tests: the per-step obligations tail block is REMOVED (#1514).

#1413 tail-injected an always-on obligations block on EVERY step. #1514 removed
that per-step injection: an outstanding obligation is only decision-relevant when
the agent tries to stop, so it is surfaced ONLY at the quiescence attempt (the
nudge — see ``tests/integration/test_wf_step.py``), never per-step.

These DB-backed tests drive the real ``compute_step_prelude`` → ``compose_step_context``
path and assert the per-step context NO LONGER renders an obligations block, even
for a child that owns an open request whose original ask has been windowed out
(AC1 + AC4 — the token-cost motivation). A session that owes nothing is unchanged
(AC3).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.harness.step_context import compose_step_context, compute_step_prelude
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.asyncio

_ACCOUNT = "acc_obl_tail"


@pytest.fixture
async def pool_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, agent_id, environment_id)`` for a fresh tenant."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'obligations-tail-test')",
                _ACCOUNT,
            )
        agent, env, _session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="obl_seed"
        )
        yield pool, _ACCOUNT, agent.id, env.id
    finally:
        await pool.close()


def _all_user_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text") or "")
    return "\n".join(parts)


@pytest.fixture
def _stub_tool_provider() -> Iterator[None]:
    prev = runtime.tool_provider
    tp = mock.Mock()
    tp.list_tools_for_session = AsyncMock(return_value=[])
    runtime.tool_provider = tp
    try:
        yield
    finally:
        runtime.tool_provider = prev


async def _prelude_and_compose(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    session_id: str,
    events: list[Any],
) -> list[dict[str, Any]]:
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    agent = await agents_service.load_for_session(pool, session, account_id=account_id)
    prelude = await compute_step_prelude(
        pool,
        session_id,
        account_id=account_id,
        session=session,
        agent=agent,
        channels=[],
        memory_store_echoes=[],
    )
    ctx = await compose_step_context(
        pool=pool,
        session=session,
        account_id=account_id,
        agent=agent,
        channels=[],
        prelude=prelude,
        events=events,  # the *windowed* slate handed to the composer
    )
    return ctx.messages


async def test_no_per_step_obligations_block_even_with_open_request(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
    _stub_tool_provider: None,
) -> None:
    """AC1/AC4: a child that owns an open request (its ask windowed out) gets NO
    per-step obligations block — the surface moved to the quiescence nudge."""
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_window"
    )
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req-windowed",
            caller={"kind": "run", "id": "run_owner"},
            depth=1,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            output_schema={"type": "object"},
            summary="summarise the dossier",
        )

    # Empty windowed slate (the original ask has scrolled out). The old #1413 block
    # would still render here; #1514 renders nothing per-step.
    messages = await _prelude_and_compose(
        pool, account_id=account_id, session_id=session.id, events=[]
    )
    joined = _all_user_text(messages)
    assert "req-windowed" not in joined, "no per-step obligations block (#1514)"
    assert "Open obligations" not in joined  # the old #1413 header is gone
    assert "summarise the dossier" not in joined


async def test_owes_request_tool_gate_still_present(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
    _stub_tool_provider: None,
) -> None:
    """The obligations are still fetched in the prelude to gate the return/error
    tool surface — removing the render must not remove the tools a session owing a
    request needs to answer it."""
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_gate"
    )
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req-gate",
            caller={"kind": "run", "id": "run_owner"},
            depth=1,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
        )
    session_basic = await sessions_service.get_session_basic(
        pool, session.id, account_id=account_id
    )
    agent = await agents_service.load_for_session(pool, session_basic, account_id=account_id)
    prelude = await compute_step_prelude(
        pool,
        session.id,
        account_id=account_id,
        session=session_basic,
        agent=agent,
        channels=[],
        memory_store_echoes=[],
    )
    tool_names = {t["function"]["name"] for t in prelude.tools if "function" in t}
    assert {"return", "error"} <= tool_names


async def test_no_obligation_block_for_session_owing_nothing(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
    _stub_tool_provider: None,
) -> None:
    """AC3: a session owing nothing is unchanged — no obligations surface at all."""
    pool, account_id, _agent_id, _env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_none"
    )
    messages = await _prelude_and_compose(
        pool, account_id=account_id, session_id=session.id, events=[]
    )
    joined = _all_user_text(messages)
    assert "Open obligations" not in joined
