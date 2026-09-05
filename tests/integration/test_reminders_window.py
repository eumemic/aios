"""The durable-reminder writer against the real windower and log.

``compose_step_context(persist_reminders=True)`` writes a reminder row once,
the next compose finds it in the window and writes nothing, and when newer
rows push it out of the window the same content is written again — the
re-emit-on-eviction policy, measured on real ``cumulative_tokens`` with the
real ``read_windowed_events``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness.concise import CONCISE_NAG_CONTENT
from aios.models.events import REMINDER_METADATA_KEY, Event
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.integration.conftest import compose_step_for, seed_agent_env_session
from tests.support import assert_message_prefix, reminder_rows

pytestmark = [pytest.mark.integration, needs_docker]

_ACCOUNT = "acc_test_stub"


@pytest.fixture
async def pool_session(
    aios_env: dict[str, str], migrated_db_url: str
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """``(pool, session_id)`` for a fresh CONCISE session under the stub account."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=_ACCOUNT,
            prefix=f"rem-window-{uuid.uuid4().hex[:6]}",
            output_style="concise",
        )
        yield pool, session.id
    finally:
        await pool.close()


async def _rows(pool: asyncpg.Pool[Any], session_id: str) -> list[Event]:
    return reminder_rows(
        await sessions_service.read_message_events(pool, session_id, account_id=_ACCOUNT)
    )


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
        self, pool_session: tuple[asyncpg.Pool[Any], str], stub_tool_provider: None
    ) -> None:
        pool, sid = pool_session
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        await _reply(pool, sid, "hi", reacting_to=1)

        ctx1, _ = await compose_step_for(pool, sid, account_id=_ACCOUNT)
        assert ctx1.reminders_written == ("concise",)
        rows = await _rows(pool, sid)
        assert [r.seq for r in rows] == [3]
        assert rows[0].data["content"] == CONCISE_NAG_CONTENT
        assert rows[0].data["metadata"][REMINDER_METADATA_KEY]["section"] == "concise"
        # The written row is the build's last message, byte-for-byte.
        assert ctx1.messages[-1] == {"role": "user", "content": CONCISE_NAG_CONTENT}

        ctx2, slate2 = await compose_step_for(pool, sid, account_id=_ACCOUNT)
        assert ctx2.reminders_written == ()
        assert ctx2.reminders_skipped == 1
        assert [r.seq for r in await _rows(pool, sid)] == [3]
        assert [e.seq for e in slate2] == [1, 2, 3]
        assert ctx1.messages == ctx2.messages

        # The session stays idle: the row is not a stimulus.
        async with pool.acquire() as conn:
            assert await queries.derive_session_status(conn, sid, account_id=_ACCOUNT) == "idle"

    async def test_preview_compose_writes_nothing_and_matches(
        self, pool_session: tuple[asyncpg.Pool[Any], str], stub_tool_provider: None
    ) -> None:
        pool, sid = pool_session
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        await _reply(pool, sid, "hi", reacting_to=1)

        preview, _ = await compose_step_for(pool, sid, account_id=_ACCOUNT, persist=False)
        assert preview.reminders_written == ("concise",)
        assert await _rows(pool, sid) == []
        sent, _ = await compose_step_for(pool, sid, account_id=_ACCOUNT)
        assert preview.messages == sent.messages


class TestEvictionReEmits:
    async def test_row_pushed_out_of_the_window_is_written_again(
        self, pool_session: tuple[asyncpg.Pool[Any], str], stub_tool_provider: None
    ) -> None:
        pool, sid = pool_session
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        await _reply(pool, sid, "hi", reacting_to=1)
        ctx1, _ = await compose_step_for(pool, sid, account_id=_ACCOUNT)
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
        ctx2, slate2 = await compose_step_for(pool, sid, account_id=_ACCOUNT, tail_budget=cum3 + 1)
        assert [e.seq for e in slate2] == [4, 5], "seqs 1-3 (incl. the row) must be evicted"
        assert ctx2.reminders_written == ("concise",), "an evicted row is re-emitted"
        rows = await _rows(pool, sid)
        assert [r.seq for r in rows] == [3, 6]
        assert rows[0].data["content"] == rows[1].data["content"]
        assert rows[0].data["metadata"] == rows[1].data["metadata"]

        # And with the new row in the window, nothing more.
        ctx3, slate3 = await compose_step_for(pool, sid, account_id=_ACCOUNT, tail_budget=cum3 + 1)
        assert 6 in [e.seq for e in slate3]
        assert ctx3.reminders_written == ()
        assert [r.seq for r in await _rows(pool, sid)] == [3, 6]
        assert_message_prefix(ctx2.messages, ctx3.messages)


class TestPrefixAcrossRealSteps:
    async def test_consecutive_composes_are_prefixes_through_the_last_assistant(
        self, pool_session: tuple[asyncpg.Pool[Any], str], stub_tool_provider: None
    ) -> None:
        pool, sid = pool_session
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        b1, _ = await compose_step_for(pool, sid, account_id=_ACCOUNT)  # writes the nag
        await _reply(pool, sid, "hi", reacting_to=1)
        b2, _ = await compose_step_for(pool, sid, account_id=_ACCOUNT)
        assert_message_prefix(b1.messages, b2.messages)
        await sessions_service.append_user_message(pool, sid, "and?", account_id=_ACCOUNT)
        b3, _ = await compose_step_for(pool, sid, account_id=_ACCOUNT)
        assert_message_prefix(b2.messages, b3.messages)
        assert b3.messages[-1]["role"] == "user"
        assert len(await _rows(pool, sid)) == 1
