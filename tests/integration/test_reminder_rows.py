"""Durable reminder rows against a real Postgres: non-stimulus by construction.

A reminder is a ``role='user'`` message row tagged
``metadata[REMINDER_METADATA_KEY]``. Pinned here, against the real
``append_event`` / sweep / windower / inbound-budget queries:

* appending one bumps NONE of ``last_stimulus_seq`` / ``last_user_seq`` /
  ``updated_at``, leaves the derived status idle and the session out of the
  wake candidates — yet the row is priced (``cumulative_tokens`` set);
* the sweep's unreacted-rows queries still return a metadata-less tool
  result past the watermark (the NULL-safety of the exclusion) and never a
  reminder row;
* the windower's retain-the-tail clamp keeps the newest STIMULUS, not the
  newest row: a reminder appended after an oversized inbound must not evict
  that inbound on a ``window_min=0`` overflow retry;
* the inbound budget does not count reminder rows.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import events as events_q
from aios.harness.context import build_messages
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.reminders import reminder_event_data
from aios.harness.sweep import (
    UNREACTED_ROWS_FLOORED_SQL,
    UNREACTED_ROWS_SQL,
    find_sessions_needing_inference,
)
from aios.services import sessions as sessions_service
from aios.services.inbound_budget import check_inbound_budget_agent
from tests.conftest import needs_docker
from tests.integration.conftest import seed_agent_env_session

pytestmark = [pytest.mark.integration, needs_docker]

_ACCOUNT = "acc_test_stub"


@pytest.fixture
async def pool_session(
    aios_env: dict[str, str], migrated_db_url: str
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """``(pool, session_id)`` for a fresh live session under the stub account."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="reminder-rows"
        )
        yield pool, session.id
    finally:
        await pool.close()


def _reminder_data(text: str = "━━━ Open obligations ━━━\n(none)") -> dict[str, Any]:
    return reminder_event_data("obligations", text)


async def _scalars(conn: asyncpg.Connection[Any], session_id: str) -> tuple[int, int, Any, int]:
    row = await conn.fetchrow(
        "SELECT last_stimulus_seq, last_user_seq, updated_at, last_reacted_seq "
        "FROM sessions WHERE id = $1",
        session_id,
    )
    assert row is not None
    return (
        row["last_stimulus_seq"],
        row["last_user_seq"],
        row["updated_at"],
        row["last_reacted_seq"],
    )


class TestReminderRowsAreNotStimuli:
    async def test_append_leaves_stimulus_scalars_untouched(
        self, pool_session: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        pool, sid = pool_session
        # A real inbound, then the assistant's reply reacting to it: idle.
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=_ACCOUNT,
                session_id=sid,
                kind="message",
                data={"role": "assistant", "content": "hi", "reacting_to": 1},
            )
            before = await _scalars(conn, sid)
            reminder = await queries.append_event(
                conn, account_id=_ACCOUNT, session_id=sid, kind="message", data=_reminder_data()
            )
            after = await _scalars(conn, sid)
            status = await queries.derive_session_status(conn, sid, account_id=_ACCOUNT)
            priced = await conn.fetchval(
                "SELECT cumulative_tokens FROM events WHERE session_id = $1 AND seq = $2",
                sid,
                reminder.seq,
            )
        assert reminder.seq == 3
        assert after == before, "a reminder row must not bump any stimulus/user scalar"
        assert status == "idle"
        assert priced is not None and priced > 0, "reminders are budgeted like any message"
        needing = await find_sessions_needing_inference(
            pool, InflightToolRegistry(), session_id=sid
        )
        assert sid not in needing

    async def test_unreacted_rows_keep_tool_results_and_drop_reminders(
        self, pool_session: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        pool, sid = pool_session
        await sessions_service.append_user_message(pool, sid, "run it", account_id=_ACCOUNT)
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=_ACCOUNT,
                session_id=sid,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "reacting_to": 1,
                    "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "bash"}}],
                },
            )
            # A metadata-LESS tool result past the watermark — the row the
            # bare `NOT (data->'metadata' ? k)` form would have lost.
            await queries.append_event(
                conn,
                account_id=_ACCOUNT,
                session_id=sid,
                kind="message",
                data={"role": "tool", "tool_call_id": "t1", "content": "ok"},
            )
            await queries.append_event(
                conn, account_id=_ACCOUNT, session_id=sid, kind="message", data=_reminder_data()
            )
            rows = await conn.fetch(UNREACTED_ROWS_SQL, [sid])
            floored = await conn.fetch(UNREACTED_ROWS_FLOORED_SQL, [sid], 0)
        for result in (rows, floored):
            roles = [r["role"] for r in result]
            assert roles == ["tool"], f"expected exactly the tool result, got {roles}"
            assert result[0]["tool_call_id"] == "t1"


class TestWindowerRetainsTheNewestStimulus:
    async def test_oversized_inbound_survives_a_trailing_reminder(
        self, pool_session: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        """An overflow retry reads the window with ``window_min=0``: the drop
        boundary snaps in ``window_max``-sized chunks and the only tail
        guarantee is the clamp. With a reminder written after the inbound,
        keying the clamp on the newest MESSAGE would keep the reminder and
        evict the inbound — the retry would then 'succeed' on a reminder-only
        prompt and never answer it. The clamp must key on the newest stimulus.
        """
        pool, sid = pool_session
        big = "lorem ipsum dolor sit amet " * 400  # well over any tiny window
        await sessions_service.append_user_message(pool, sid, big, account_id=_ACCOUNT)
        async with pool.acquire() as conn:
            await queries.append_event(
                conn, account_id=_ACCOUNT, session_id=sid, kind="message", data=_reminder_data()
            )
            cums = await conn.fetch(
                "SELECT seq, cumulative_tokens FROM events WHERE session_id = $1 "
                "AND kind = 'message' ORDER BY seq",
                sid,
            )
            cum_user, cum_rem = cums[0]["cumulative_tokens"], cums[1]["cumulative_tokens"]
            reminder_tokens = cum_rem - cum_user
            # Size the window so a window_min=0 chunk snap lands the boundary
            # INSIDE the inbound: total mod window_max <= the reminder's price.
            window_max = cum_rem - reminder_tokens + 1
            windowed = await events_q.read_windowed_events(
                conn,
                sid,
                account_id=_ACCOUNT,
                window_min=0,
                window_max=window_max,
                model=f"fake/rem-{uuid.uuid4().hex[:8]}",
                overhead_local=0,
            )
        seqs = [e.seq for e in windowed.events]
        assert 1 in seqs, f"the unanswered inbound was evicted; retained seqs={seqs}"
        assert 2 in seqs
        ctx = build_messages(windowed.events, system_prompt=None, omission=windowed.omission)
        assert any("lorem ipsum" in str(m.get("content")) for m in ctx.messages)
        # The reminder is not a stimulus: the build reacts to the inbound only.
        assert ctx.reacting_to == 1


class TestReminderRowsArePricedBare:
    async def test_cumulative_tokens_is_bare_render_plus_separator(
        self, pool_session: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        """A reminder's stored delta is its bare render plus the adjacent-user
        separator pre-pay — the same pricing path as any user row, minus the
        ``[received=…]`` envelope that path adds to a real inbound."""
        from datetime import UTC, datetime

        from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT, render_user_event
        from aios.harness.tokens import approx_tokens

        pool, sid = pool_session
        text = "━━━ Open obligations ━━━\n• req_01 [self] opened=2026-09-05T00:00:00Z task"
        data = _reminder_data(text)
        await sessions_service.append_user_message(pool, sid, "hello", account_id=_ACCOUNT)
        async with pool.acquire() as conn:
            await queries.append_event(
                conn, account_id=_ACCOUNT, session_id=sid, kind="message", data=data
            )
            cums = await conn.fetch(
                "SELECT cumulative_tokens FROM events WHERE session_id = $1 "
                "AND kind = 'message' ORDER BY seq",
                sid,
            )
        stored_delta = cums[1]["cumulative_tokens"] - cums[0]["cumulative_tokens"]
        bare = {"role": "user", "content": text}
        expected = approx_tokens(
            [bare, {"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT}]
        )
        assert stored_delta == expected
        assert render_user_event(data, None, None, datetime.now(UTC)) == bare


class TestInboundBudgetIgnoresReminders:
    async def test_reminder_rows_do_not_consume_the_agent_budget(
        self, pool_session: tuple[asyncpg.Pool[Any], str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pool, sid = pool_session
        monkeypatch.setenv("AIOS_INBOUND_RATE_AGENT_MAX_PER_WINDOW", "2")
        monkeypatch.setenv("AIOS_INBOUND_RATE_AGENT_WINDOW_SECONDS", "3600")
        get_settings.cache_clear()
        try:
            assert get_settings().inbound_rate_agent_max_per_window == 2
            async with pool.acquire() as conn:
                for _ in range(3):
                    await queries.append_event(
                        conn,
                        account_id=_ACCOUNT,
                        session_id=sid,
                        kind="message",
                        data=_reminder_data(),
                    )
            assert (
                await check_inbound_budget_agent(pool, account_id=_ACCOUNT, session_id=sid) is True
            ), "three reminder rows must not exhaust a budget of two"
            for text in ("one", "two"):
                await sessions_service.append_user_message(pool, sid, text, account_id=_ACCOUNT)
            assert (
                await check_inbound_budget_agent(pool, account_id=_ACCOUNT, session_id=sid) is False
            ), "two real inbounds DO exhaust a budget of two"
        finally:
            get_settings.cache_clear()
