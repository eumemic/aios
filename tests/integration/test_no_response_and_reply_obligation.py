"""``no_response`` and the reply-or-decline obligation against a real Postgres (#2092).

* A quiet tool result (``no_response``'s) closes its call but is not a stimulus:
  it leaves ``last_stimulus_seq`` alone, the session idles, and the sweep never
  wakes it — while an ordinary tool result in the same position does.
* The quiescence guard nudges exactly once for an unanswered inbound channel
  message, then lets the session idle and records a ``reply_lapsed`` span.
* Callers cannot mint the harness's metadata markers on a user message.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ValidationError
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.reply_obligation import (
    NO_RESPONSE_TOOL_NAME,
    REPLY_LAPSED_SPAN_EVENT,
    REPLY_NUDGE_METADATA_KEY,
    find_reply_due,
)
from aios.harness.sweep import UNREACTED_ROWS_SQL, find_sessions_needing_inference
from aios.models.events import QUIET_RESULT_METADATA_KEY, REMINDER_METADATA_KEY
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.integration.conftest import seed_agent_env_session

pytestmark = [pytest.mark.integration, needs_docker]

_ACCOUNT = "acc_test_stub"
_CHANNEL = "signal/bot/chat-a"


@pytest.fixture
async def pool_session(
    aios_env: dict[str, str], migrated_db_url: str
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="no-response"
        )
        yield pool, session.id
    finally:
        await pool.close()


async def _scalars(conn: asyncpg.Connection[Any], sid: str) -> dict[str, int]:
    row = await conn.fetchrow(
        "SELECT last_stimulus_seq, last_reacted_seq, open_tool_call_count "
        "FROM sessions WHERE id = $1",
        sid,
    )
    assert row is not None
    return dict(row)


async def _inbound(pool: asyncpg.Pool[Any], sid: str, text: str = "are you there?") -> int:
    event = await sessions_service.append_user_message(
        pool, sid, text, metadata={"channel": _CHANNEL}, account_id=_ACCOUNT
    )
    return event.seq


async def _assistant_calling(
    conn: asyncpg.Connection[Any], sid: str, name: str, *, reacting_to: int
) -> None:
    await queries.append_event(
        conn,
        account_id=_ACCOUNT,
        session_id=sid,
        kind="message",
        data={
            "role": "assistant",
            "content": "",
            "reacting_to": reacting_to,
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": name, "arguments": "{}"}}
            ],
        },
    )


class TestQuietResult:
    @pytest.mark.parametrize("quiet", [True, False])
    async def test_quiet_result_closes_the_call_without_waking(
        self, pool_session: tuple[asyncpg.Pool[Any], str], quiet: bool
    ) -> None:
        pool, sid = pool_session
        inbound = await _inbound(pool, sid)
        result: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": "t1",
            "name": NO_RESPONSE_TOOL_NAME if quiet else "bash",
            "content": "Nothing sent." if quiet else "ok",
        }
        if quiet:
            result["metadata"] = {QUIET_RESULT_METADATA_KEY: {"tool": NO_RESPONSE_TOOL_NAME}}
        async with pool.acquire() as conn:
            await _assistant_calling(
                conn, sid, NO_RESPONSE_TOOL_NAME if quiet else "bash", reacting_to=inbound
            )
            before = await _scalars(conn, sid)
            assert before["open_tool_call_count"] == 1
            await queries.append_event(
                conn, account_id=_ACCOUNT, session_id=sid, kind="message", data=result
            )
            after = await _scalars(conn, sid)
            status = await queries.derive_session_status(conn, sid, account_id=_ACCOUNT)
            unreacted = await conn.fetch(UNREACTED_ROWS_SQL, [sid])
        needing = await find_sessions_needing_inference(
            pool, InflightToolRegistry(), session_id=sid
        )
        assert after["open_tool_call_count"] == 0, "the result always closes its call"
        if quiet:
            assert after["last_stimulus_seq"] == before["last_stimulus_seq"]
            assert status == "idle"
            assert unreacted == []
            assert sid not in needing
        else:
            assert after["last_stimulus_seq"] > before["last_stimulus_seq"]
            assert status == "active"
            assert [r["role"] for r in unreacted] == ["tool"]
            assert sid in needing


class TestReplyNudge:
    async def _bare_text_turn(self, pool: asyncpg.Pool[Any], sid: str) -> Any:
        """One tool-call-free assistant turn, reply obligation computed from the log
        the way the loop computes it from the step's slate."""
        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, sid, account_id=_ACCOUNT)
            reacting_to = (await _scalars(conn, sid))["last_stimulus_seq"]
        due = find_reply_due(
            events, reacting_to=reacting_to, connector_tool_names=frozenset({"signal_send"})
        )
        return await sessions_service.append_assistant_and_guard_quiescence(
            pool,
            sid,
            {
                "role": "assistant",
                "content": "I think they want a summary.",
                "reacting_to": reacting_to,
            },
            account_id=_ACCOUNT,
            reply_due=due,
        )

    async def _rows(self, pool: asyncpg.Pool[Any], sid: str) -> tuple[list[Any], list[Any]]:
        async with pool.acquire() as conn:
            nudges = await conn.fetch(
                "SELECT seq, data FROM events WHERE session_id = $1 AND kind = 'message' "
                "AND data->'metadata' ? $2 ORDER BY seq",
                sid,
                REPLY_NUDGE_METADATA_KEY,
            )
            lapses = await conn.fetch(
                "SELECT seq, data FROM events WHERE session_id = $1 AND kind = 'span' "
                "AND data->>'event' = $2 ORDER BY seq",
                sid,
                REPLY_LAPSED_SPAN_EVENT,
            )
        return list(nudges), list(lapses)

    async def test_one_nudge_then_lapse(self, pool_session: tuple[asyncpg.Pool[Any], str]) -> None:
        pool, sid = pool_session
        inbound = await _inbound(pool, sid)

        first = await self._bare_text_turn(pool, sid)
        assert first.nudged is True
        nudges, lapses = await self._rows(pool, sid)
        assert len(nudges) == 1 and lapses == []
        assert nudges[0]["data"]["metadata"][REPLY_NUDGE_METADATA_KEY]["through_seq"] == inbound
        async with pool.acquire() as conn:
            assert await queries.derive_session_status(conn, sid, account_id=_ACCOUNT) == "active"

        second = await self._bare_text_turn(pool, sid)
        assert second.nudged is False
        nudges, lapses = await self._rows(pool, sid)
        assert len(nudges) == 1, "the same inbound is never nudged twice"
        assert len(lapses) == 1
        assert lapses[0]["data"]["through_seq"] == inbound
        assert lapses[0]["data"]["channels"] == [_CHANNEL]
        async with pool.acquire() as conn:
            assert await queries.derive_session_status(conn, sid, account_id=_ACCOUNT) == "idle"

    async def test_a_new_inbound_after_the_lapse_gets_its_own_nudge(
        self, pool_session: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        pool, sid = pool_session
        await _inbound(pool, sid)
        await self._bare_text_turn(pool, sid)
        await self._bare_text_turn(pool, sid)
        newer = await _inbound(pool, sid, "hello again")
        third = await self._bare_text_turn(pool, sid)
        assert third.nudged is True
        nudges, _ = await self._rows(pool, sid)
        assert [n["data"]["metadata"][REPLY_NUDGE_METADATA_KEY]["through_seq"] for n in nudges][
            -1
        ] == newer

    async def test_a_reply_owes_nothing(self, pool_session: tuple[asyncpg.Pool[Any], str]) -> None:
        pool, sid = pool_session
        inbound = await _inbound(pool, sid)
        async with pool.acquire() as conn:
            await queries.set_session_focal_channel(conn, sid, _CHANNEL, account_id=_ACCOUNT)
            await _assistant_calling(conn, sid, "signal_send", reacting_to=inbound)
            await queries.append_event(
                conn,
                account_id=_ACCOUNT,
                session_id=sid,
                kind="message",
                data={"role": "tool", "tool_call_id": "t1", "content": '{"sent": true}'},
            )
        result = await self._bare_text_turn(pool, sid)
        assert result.nudged is False
        nudges, lapses = await self._rows(pool, sid)
        assert nudges == [] and lapses == []


class TestReservedMetadata:
    @pytest.mark.parametrize(
        "key", [REMINDER_METADATA_KEY, QUIET_RESULT_METADATA_KEY, REPLY_NUDGE_METADATA_KEY]
    )
    async def test_callers_cannot_mint_harness_markers(
        self, pool_session: tuple[asyncpg.Pool[Any], str], key: str
    ) -> None:
        pool, sid = pool_session
        with pytest.raises(ValidationError, match=key):
            await sessions_service.append_user_message(
                pool, sid, "sneaky", metadata={key: {"through_seq": 10**9}}, account_id=_ACCOUNT
            )
