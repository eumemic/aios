"""Integration tests for the always-on obligations tail block (#1413).

DB-backed (testcontainer Postgres). The block's whole reason to exist is to
survive context-window erasure of the original request user message: a long-lived
child whose opening request scrolled out of the window must STILL see, every step,
the request_id it owes a response to. These tests drive the real
``compute_step_prelude`` → ``compose_step_context`` path and assert:

* a child that owns an open request gets the obligation rendered as the FINAL
  user-role message — even when the original request message has been windowed out;
* the rendered line names the literal request_id (what ``return(request_id=...)``
  needs) and the caller origin;
* an ordinary session that owes nothing gets no obligations block;
* answering the request drops the block on the next compose (no stale reminder).
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


def _final_user_text(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        return blk.get("text")
    return None


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


async def test_obligation_renders_when_original_request_is_windowed_out(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
    _stub_tool_provider: None,
) -> None:
    """The repro: the opening request message is NOT in the windowed slate, yet the
    obligation block still names the owed request_id as the final user line."""
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
            summary="summarise the dossier",
        )

    # Hand the composer an EMPTY windowed slate — i.e. the original request user
    # message has scrolled out. The block is rebuilt from prelude.obligations, which
    # reads the full log, so it must still appear.
    messages = await _prelude_and_compose(
        pool, account_id=account_id, session_id=session.id, events=[]
    )
    final = _final_user_text(messages)
    assert final is not None
    assert "req-windowed" in final, "owed request_id must survive windowing erasure"
    assert "summarise the dossier" in final
    assert "[run]" in final


async def test_no_obligation_block_for_session_owing_nothing(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
    _stub_tool_provider: None,
) -> None:
    pool, account_id, _agent_id, _env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_none"
    )
    messages = await _prelude_and_compose(
        pool, account_id=account_id, session_id=session.id, events=[]
    )
    joined = "\n".join(
        m["content"]
        for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    )
    assert "request_id" not in joined.lower() or "req-" not in joined


async def test_answering_drops_the_block_on_next_compose(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
    _stub_tool_provider: None,
) -> None:
    pool, account_id, _agent_id, env_id = pool_env
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix="obl_drop"
    )
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=account_id,
            request_id="req-drop",
            caller={"kind": "session", "id": "ses_owner"},
            depth=0,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            summary="do work",
        )
    before = await _prelude_and_compose(
        pool, account_id=account_id, session_id=session.id, events=[]
    )
    assert "req-drop" in (_final_user_text(before) or "")

    async with pool.acquire() as conn:
        await queries.write_response_if_absent(
            conn,
            session.id,
            account_id=account_id,
            request_id="req-drop",
            is_error=False,
            result={"ok": True},
            error=None,
        )

    after = await _prelude_and_compose(
        pool, account_id=account_id, session_id=session.id, events=[]
    )
    assert "req-drop" not in (_final_user_text(after) or "")
