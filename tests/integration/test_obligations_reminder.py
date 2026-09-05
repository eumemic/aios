"""The obligations reminder's presence gate against the real log.

An obligation whose ORIGINAL ask is still in the windowed slate earns no
reminder row (the ask itself is the stimulus, intact above); one whose ask
has scrolled out earns the listing, once; when the open set empties while a
listing is still in the window, the one-line supersession is written once;
and a later obligation whose ask is present writes nothing again.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness.obligations import OBLIGATIONS_EMPTY_CONTENT
from aios.harness.step_context import StepContext
from aios.models.events import Event
from aios.models.sessions import Ok
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.integration.conftest import compose_step_for, seed_agent_env_session
from tests.support import reminder_rows

pytestmark = [pytest.mark.integration, needs_docker]

_ACCOUNT = "acc_test_stub"


@pytest.fixture
async def pool_session(
    aios_env: dict[str, str], migrated_db_url: str
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """``(pool, session_id, environment_id)`` for a fresh live session."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        _agent, env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix=f"obl-rem-{uuid.uuid4().hex[:6]}"
        )
        yield pool, session.id, env.id
    finally:
        await pool.close()


async def _open(
    pool: asyncpg.Pool[Any], session_id: str, env_id: str, request_id: str, *, ask: str
) -> None:
    """Open an awaited obligation the way a caller does: the request user
    message (stamped with the request_id) plus the ``request_opened`` edge."""
    await sessions_service.append_user_message(
        pool,
        session_id,
        ask,
        account_id=_ACCOUNT,
        metadata={"request": {"request_id": request_id}},
    )
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            caller={"kind": "run", "id": "run_owner"},
            depth=1,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            summary=ask,
        )


async def _answer(pool: asyncpg.Pool[Any], session_id: str, request_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.write_response_if_absent(
            conn,
            session_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            outcome=Ok(result={"ok": True}),
        )


async def _compose(pool: asyncpg.Pool[Any], session_id: str, *, events: list[Event]) -> StepContext:
    """Compose over a caller-chosen slate (the windower's output is simulated
    so an ask can be 'scrolled out' without a real overflow)."""
    ctx, _ = await compose_step_for(pool, session_id, account_id=_ACCOUNT, events=events)
    return ctx


async def _log(pool: asyncpg.Pool[Any], session_id: str) -> list[Event]:
    return await sessions_service.read_message_events(pool, session_id, account_id=_ACCOUNT)


async def _rows(pool: asyncpg.Pool[Any], session_id: str) -> list[Event]:
    return reminder_rows(await _log(pool, session_id))


class TestPresenceGate:
    async def test_ask_in_window_writes_no_row(
        self, pool_session: tuple[asyncpg.Pool[Any], str, str], stub_tool_provider: None
    ) -> None:
        pool, sid, env_id = pool_session
        await _open(pool, sid, env_id, "req-present", ask="summarise the dossier")
        ctx = await _compose(pool, sid, events=await _log(pool, sid))
        assert ctx.reminders_written == ()
        assert ctx.reminders_skipped == 1
        assert await _rows(pool, sid) == []
        # The ask itself is the final user content — the real stimulus.
        assert "summarise the dossier" in str(ctx.messages[-1]["content"])

    async def test_ask_scrolled_out_writes_the_listing_once(
        self, pool_session: tuple[asyncpg.Pool[Any], str, str], stub_tool_provider: None
    ) -> None:
        pool, sid, env_id = pool_session
        await _open(pool, sid, env_id, "req-gone", ask="summarise the dossier")
        # The ask has scrolled out: an empty slate.
        ctx = await _compose(pool, sid, events=[])
        assert ctx.reminders_written == ("obligations",)
        rows = await _rows(pool, sid)
        assert len(rows) == 1
        assert "req-gone" in rows[0].data["content"]
        assert "[run]" in rows[0].data["content"]
        assert ctx.messages[-1] == {"role": "user", "content": rows[0].data["content"]}
        # The next compose sees the row in its window: no second write.
        ctx2 = await _compose(pool, sid, events=rows)
        assert ctx2.reminders_written == ()
        assert len(await _rows(pool, sid)) == 1

    async def test_emptied_then_new_present_ask(
        self, pool_session: tuple[asyncpg.Pool[Any], str, str], stub_tool_provider: None
    ) -> None:
        pool, sid, env_id = pool_session
        await _open(pool, sid, env_id, "req-1", ask="first task")
        ctx1 = await _compose(pool, sid, events=[])  # ask scrolled out → listing
        assert ctx1.reminders_written == ("obligations",)
        listing = (await _rows(pool, sid))[0]

        # Answered while the listing is still in the window: the one-liner,
        # exactly once.
        await _answer(pool, sid, "req-1")
        ctx2 = await _compose(pool, sid, events=[listing])
        assert ctx2.reminders_written == ("obligations",)
        rows = await _rows(pool, sid)
        assert [r.data["content"] for r in rows][-1] == OBLIGATIONS_EMPTY_CONTENT
        ctx3 = await _compose(pool, sid, events=rows)
        assert ctx3.reminders_written == ()
        assert len(await _rows(pool, sid)) == 2

        # A new obligation whose ask is in the window: presence-gated, no row.
        await _open(pool, sid, env_id, "req-2", ask="second task")
        ctx4 = await _compose(pool, sid, events=await _log(pool, sid))
        assert ctx4.reminders_written == ()
        assert len(await _rows(pool, sid)) == 2

    async def test_never_owed_and_emptied_without_listing_write_nothing(
        self, pool_session: tuple[asyncpg.Pool[Any], str, str], stub_tool_provider: None
    ) -> None:
        pool, sid, env_id = pool_session
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        assert (await _compose(pool, sid, events=await _log(pool, sid))).reminders_written == ()
        # Opened and answered with no listing ever written: still nothing.
        await _open(pool, sid, env_id, "req-quick", ask="quick one")
        await _answer(pool, sid, "req-quick")
        assert (await _compose(pool, sid, events=[])).reminders_written == ()
        assert await _rows(pool, sid) == []
