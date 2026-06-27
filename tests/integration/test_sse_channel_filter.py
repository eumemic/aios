"""Integration tests: ``sse_event_stream`` honors the channel filter on both
backfill and live tail, while NULL-channel terminal/lifecycle events always
reach the consumer (#1613).

DB-backed (testcontainer Postgres). Drives the backfill path directly and
asserts the channel-scoped SELECT excludes cross-channel message rows but still
yields a terminal lifecycle event.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.api.sse import sse_event_stream
from aios.db import queries
from aios.db.listen import open_listen_for_events
from aios.db.pool import create_pool
from aios.db.queries import sessions as session_queries
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

CHAN_A = "signal/bot/6c21718f-f095-483f-8cd6-610137d581aa"  # DM (UUID)
CHAN_B = "signal/bot/abcDEF123_-=="  # group (base64)


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_sse_filt', NULL, TRUE, 'sse-filter-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_sse_filt", prefix="sse-filter-test"
        )
        yield pool, migrated_db_url, "acc_sse_filt", session.id
    finally:
        await pool.close()


async def _inbound(
    conn: asyncpg.Connection[Any], account_id: str, session_id: str, channel: str, text: str
) -> None:
    await queries.append_event(
        conn,
        account_id=account_id,
        session_id=session_id,
        kind="message",
        data={"role": "user", "content": text, "metadata": {"channel": channel}},
        orig_channel=channel,
    )


async def _drain_backfill(
    db_url: str, pool: asyncpg.Pool[Any], session_id: str, **kwargs: Any
) -> list[dict[str, Any]]:
    """Run ``sse_event_stream`` over the backfill, collecting message-event
    payloads until the ``done`` sentinel (terminal lifecycle) arrives."""
    subscription = await open_listen_for_events(db_url, session_id)
    out: list[dict[str, Any]] = []
    gen = sse_event_stream(subscription, pool, session_id, **kwargs)
    async for sse in gen:
        if sse.event == "done":
            out.append({"__done__": True})
            break
        if sse.event == "event":
            out.append(json.loads(str(sse.data)))
    await gen.aclose()  # type: ignore[attr-defined]
    return out


async def test_backfill_channel_filter_excludes_cross_channel(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """Backfill with ?channel=A yields the A inbound + the terminal lifecycle,
    and EXCLUDES the cross-channel B inbound."""
    pool, db_url, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await session_queries.set_session_focal_channel(
            conn, session_id, CHAN_A, account_id=account_id
        )
        await _inbound(conn, account_id, session_id, CHAN_A, "from A")
        await _inbound(conn, account_id, session_id, CHAN_B, "from B")
        # Terminal lifecycle event (NULL channel) must still reach the consumer.
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="lifecycle",
            data={"status": "terminated"},
            orig_channel=None,
        )

    payloads = await _drain_backfill(db_url, pool, session_id, channels=[CHAN_A])
    contents = [p.get("data", {}).get("content") for p in payloads if "data" in p]
    assert "from A" in contents
    assert "from B" not in contents  # cross-channel exclusion (the bug)
    # The terminal lifecycle (NULL channel) still reached the consumer.
    assert {"__done__": True} in payloads
    # Every message row that came through is channel A.
    msg_channels = {p["channel"] for p in payloads if p.get("kind") == "message"}
    assert msg_channels == {CHAN_A}


async def test_backfill_no_filter_unchanged(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """Regression: no filter ⇒ both channels' inbound + terminal all delivered."""
    pool, db_url, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await _inbound(conn, account_id, session_id, CHAN_A, "from A")
        await _inbound(conn, account_id, session_id, CHAN_B, "from B")
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="lifecycle",
            data={"status": "terminated"},
            orig_channel=None,
        )

    payloads = await _drain_backfill(db_url, pool, session_id)
    contents = [p.get("data", {}).get("content") for p in payloads if "data" in p]
    assert "from A" in contents
    assert "from B" in contents
    assert {"__done__": True} in payloads


async def test_backfill_chat_type_filter(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """?chat_type=dm yields only the DM (CHAN_A) message; the group inbound is
    excluded, terminal still delivered."""
    pool, db_url, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await _inbound(conn, account_id, session_id, CHAN_A, "dm msg")
        await _inbound(conn, account_id, session_id, CHAN_B, "group msg")
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="lifecycle",
            data={"status": "terminated"},
            orig_channel=None,
        )

    payloads = await _drain_backfill(db_url, pool, session_id, chat_type="dm")
    contents = [p.get("data", {}).get("content") for p in payloads if "data" in p]
    assert "dm msg" in contents
    assert "group msg" not in contents
    assert {"__done__": True} in payloads
