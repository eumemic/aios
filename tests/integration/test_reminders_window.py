"""The durable-reminder writer against the real windower and log.

``compose_step_context(persist_reminders=True)`` writes a reminder row once,
the next compose finds it in the window and writes nothing, and when newer
rows push it out of the window the same content is written again — the
re-emit-on-eviction policy, measured on real ``cumulative_tokens`` with the
real ``read_windowed_events``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.harness.concise import CONCISE_NAG_CONTENT
from aios.harness.step_context import (
    StepContext,
    compose_step_context,
    compute_step_prelude,
    prelude_overhead_local,
)
from aios.models.events import REMINDER_METADATA_KEY, Event, is_reminder_event
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.support import assert_message_prefix

pytestmark = [pytest.mark.integration, needs_docker]

_ACCOUNT = "acc_test_stub"


@pytest.fixture
async def pool_session(
    aios_env: dict[str, str], migrated_db_url: str
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """``(pool, session_id)`` for a fresh CONCISE session under the stub account."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        prefix = f"rem-window-{uuid.uuid4().hex[:6]}"
        # Sessions pin their agent version at creation, so the style is set
        # on the agent BEFORE the session exists (not via update_agent).
        agent = await agents_service.create_agent(
            pool,
            account_id=_ACCOUNT,
            name=f"{prefix}-agent",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            output_style="concise",
        )
        env = await environments_service.create_environment(
            pool, account_id=_ACCOUNT, name=f"{prefix}-env"
        )
        session = await sessions_service.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title=None,
            metadata={},
            account_id=_ACCOUNT,
        )
        yield pool, session.id
    finally:
        await pool.close()


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


async def _compose(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    tail_budget: int | None = None,
    persist: bool = True,
) -> tuple[StepContext, list[Event]]:
    """Prelude → real windowed read → compose, the step's own sequence.

    ``tail_budget`` sizes the window for the EVENTS alone (the prelude's
    overhead is added back so the windower's budget is exactly this many
    tokens of log); ``None`` keeps the agent's generous window.
    """
    session = await sessions_service.get_session_basic(pool, session_id, account_id=_ACCOUNT)
    agent = await agents_service.load_for_session(pool, session, account_id=_ACCOUNT)
    prelude = await compute_step_prelude(
        pool,
        session_id,
        account_id=_ACCOUNT,
        session=session,
        agent=agent,
        channels=[],
        memory_store_echoes=[],
    )
    overhead = prelude_overhead_local(prelude)
    windowed = await sessions_service.read_windowed_events(
        pool,
        session_id,
        account_id=_ACCOUNT,
        window_min=1 if tail_budget is not None else agent.window_min,
        window_max=overhead.total + tail_budget if tail_budget is not None else agent.window_max,
        model=agent.model,
        overhead_local=overhead,
    )
    ctx = await compose_step_context(
        pool=pool,
        session=session,
        account_id=_ACCOUNT,
        agent=agent,
        channels=[],
        prelude=prelude,
        events=windowed.events,
        omission=windowed.omission,
        persist_reminders=persist,
    )
    return ctx, windowed.events


async def _reminder_rows(pool: asyncpg.Pool[Any], session_id: str) -> list[Event]:
    rows = await sessions_service.read_message_events(pool, session_id, account_id=_ACCOUNT)
    return [e for e in rows if is_reminder_event(e.kind, e.data)]


async def _cumulative(pool: asyncpg.Pool[Any], session_id: str, seq: int) -> int:
    async with pool.acquire() as conn:
        cum = await conn.fetchval(
            "SELECT cumulative_tokens FROM events WHERE session_id = $1 AND seq = $2",
            session_id,
            seq,
        )
    assert isinstance(cum, int)
    return cum


async def _reply(
    pool: asyncpg.Pool[Any], session_id: str, content: str, *, reacting_to: int
) -> int:
    async with pool.acquire() as conn:
        row = await queries.append_event(
            conn,
            account_id=_ACCOUNT,
            session_id=session_id,
            kind="message",
            data={"role": "assistant", "content": content, "reacting_to": reacting_to},
        )
    return row.seq


class TestWriteOnceThenReplay:
    async def test_first_compose_writes_next_compose_replays(
        self, pool_session: tuple[asyncpg.Pool[Any], str], _stub_tool_provider: None
    ) -> None:
        pool, sid = pool_session
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        await _reply(pool, sid, "hi", reacting_to=1)

        ctx1, _ = await _compose(pool, sid)
        assert ctx1.reminders_written == ("concise",)
        rows = await _reminder_rows(pool, sid)
        assert [r.seq for r in rows] == [3]
        assert rows[0].data["content"] == CONCISE_NAG_CONTENT
        assert rows[0].data["metadata"][REMINDER_METADATA_KEY]["section"] == "concise"
        # The written row is the build's last message, byte-for-byte.
        assert ctx1.messages[-1] == {"role": "user", "content": CONCISE_NAG_CONTENT}

        ctx2, slate2 = await _compose(pool, sid)
        assert ctx2.reminders_written == ()
        assert ctx2.reminders_skipped == 1
        assert [r.seq for r in await _reminder_rows(pool, sid)] == [3]
        assert [e.seq for e in slate2] == [1, 2, 3]
        assert ctx1.messages == ctx2.messages

        # The session stays idle: the row is not a stimulus.
        async with pool.acquire() as conn:
            assert await queries.derive_session_status(conn, sid, account_id=_ACCOUNT) == "idle"

    async def test_preview_compose_writes_nothing_and_matches(
        self, pool_session: tuple[asyncpg.Pool[Any], str], _stub_tool_provider: None
    ) -> None:
        pool, sid = pool_session
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        await _reply(pool, sid, "hi", reacting_to=1)

        preview, _ = await _compose(pool, sid, persist=False)
        assert preview.reminders_written == ("concise",)
        assert await _reminder_rows(pool, sid) == []
        sent, _ = await _compose(pool, sid, persist=True)
        assert preview.messages == sent.messages


class TestEvictionReEmits:
    async def test_row_pushed_out_of_the_window_is_written_again(
        self, pool_session: tuple[asyncpg.Pool[Any], str], _stub_tool_provider: None
    ) -> None:
        pool, sid = pool_session
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        await _reply(pool, sid, "hi", reacting_to=1)
        ctx1, _ = await _compose(pool, sid)
        assert ctx1.reminders_written == ("concise",)  # seq 3

        # Newer traffic after the row: a fat inbound and its reply.
        await sessions_service.append_user_message(
            pool, sid, "more please " * 300, account_id=_ACCOUNT
        )  # seq 4
        await _reply(pool, sid, "sure", reacting_to=4)  # seq 5

        # Size the events budget just past the row's cumulative price: the
        # chunked snap then drops at least through seq 3, while the windower's
        # newest-stimulus clamp keeps the fat inbound (seq 4) and its reply.
        cum3 = await _cumulative(pool, sid, 3)
        ctx2, slate2 = await _compose(pool, sid, tail_budget=cum3 + 1)
        assert [e.seq for e in slate2] == [4, 5], "seqs 1-3 (incl. the row) must be evicted"
        assert ctx2.reminders_written == ("concise",), "an evicted row is re-emitted"
        rows = await _reminder_rows(pool, sid)
        assert [r.seq for r in rows] == [3, 6]
        assert rows[0].data["content"] == rows[1].data["content"]
        assert rows[0].data["metadata"] == rows[1].data["metadata"]

        # And with the new row in the window, nothing more.
        ctx3, slate3 = await _compose(pool, sid, tail_budget=cum3 + 1)
        assert 6 in [e.seq for e in slate3]
        assert ctx3.reminders_written == ()
        assert [r.seq for r in await _reminder_rows(pool, sid)] == [3, 6]
        assert_message_prefix(ctx2.messages, ctx3.messages)


class TestPrefixAcrossRealSteps:
    async def test_consecutive_composes_are_prefixes_through_the_last_assistant(
        self, pool_session: tuple[asyncpg.Pool[Any], str], _stub_tool_provider: None
    ) -> None:
        pool, sid = pool_session
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        b1, _ = await _compose(pool, sid)  # writes the nag (merged into the inbound)
        await _reply(pool, sid, "hi", reacting_to=1)
        b2, _ = await _compose(pool, sid)
        assert_message_prefix(b1.messages, b2.messages)
        await sessions_service.append_user_message(pool, sid, "and?", account_id=_ACCOUNT)
        b3, _ = await _compose(pool, sid)
        assert_message_prefix(b2.messages, b3.messages)
        assert b3.messages[-1]["role"] == "user"
        assert len(await _reminder_rows(pool, sid)) == 1
