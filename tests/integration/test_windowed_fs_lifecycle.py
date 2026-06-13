"""Integration: the windowing read carries model-visible FS-loss notices.

``read_windowed_events`` feeds ``build_messages``, which renders the §5.9
sandbox FS-loss notices (``sandbox_fs_reset`` / ``_expired`` /
``_over_limit``, appended as ``kind='lifecycle'``). But every windowing
read path filtered ``kind='message'``, so the notices were stripped at the
SQL layer and never reached the model — the feature was dead in
production. These tests pin that the allowlisted lifecycle events survive
the read, and that they window out by *seq* (not the token boundary, which
they have no part in — they carry NULL ``cumulative_tokens``) so a notice
scrolls out of context alongside the messages around its reset point.

The bug lives in the SQL kind filter, so it is not unit-reachable: the
``read_windowed_events`` unit FakeConn dispatches on SQL text and never
models real rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness.context import build_messages
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_fsnotice', NULL, TRUE, 'fsnotice-test')"
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_fsnotice", prefix="fsnotice-test"
        )
        yield pool, "acc_fsnotice", session.id
    finally:
        await pool.close()


async def _msg(
    conn: asyncpg.Connection[Any], account_id: str, session_id: str, content: str
) -> Any:
    return await queries.append_event(
        conn,
        account_id=account_id,
        session_id=session_id,
        kind="message",
        data={"role": "user", "content": content},
    )


async def _notice(
    conn: asyncpg.Connection[Any], account_id: str, session_id: str, reason: str
) -> Any:
    return await queries.append_event(
        conn,
        account_id=account_id,
        session_id=session_id,
        kind="lifecycle",
        data={"event": "sandbox_fs_reset", "reason": reason},
    )


async def test_windowed_read_includes_and_renders_fs_notice(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Small session (everything fits → full-log path): a sandbox_fs_reset
    notice survives the windowing read, and build_messages renders it as a
    user-role notice the model can act on."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await _msg(conn, account_id, session_id, "install numpy")
        await _notice(conn, account_id, session_id, "snapshot_missing")

        windowed = await queries.read_windowed_events(
            conn,
            session_id,
            account_id=account_id,
            window_min=1_000,
            window_max=8_000,
            model="openrouter/test",
            overhead_local=0,
        )

    # The lifecycle notice is present in the windowed events (RED pre-fix:
    # stripped by the kind='message' filter).
    assert any(
        e.kind == "lifecycle" and e.data.get("event") == "sandbox_fs_reset" for e in windowed.events
    )

    # End-to-end: build_messages renders it as a user-role notice.
    result = build_messages(windowed.events, system_prompt=None)
    assert any(
        m["role"] == "user"
        and isinstance(m.get("content"), str)
        and "fresh base filesystem" in m["content"]
        for m in result.messages
    )


async def test_windowed_context_events_seq_bounds_notices(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Range-scan path: notices window out by seq with their surrounding
    messages. A notice in the dropped prefix is excluded; one in the
    retained window is kept. Driven by calling read_windowed_context_events
    with an explicit drop so the bound is exact, not token-math-dependent."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        # Log order: m1, notice_a, m2, m3, notice_b, m4.
        m1 = await _msg(conn, account_id, session_id, "one")
        notice_a = await _notice(conn, account_id, session_id, "snapshot_missing")
        m2 = await _msg(conn, account_id, session_id, "two")
        m3 = await _msg(conn, account_id, session_id, "three")
        notice_b = await _notice(conn, account_id, session_id, "environment_image_changed")
        m4 = await _msg(conn, account_id, session_id, "four")

        # Drop boundary at m2's cumulative_tokens: messages with
        # cumulative_tokens <= drop (m1, m2) are dropped; m3, m4 retained.
        # max dropped-message seq is m2.seq.
        drop_row = await conn.fetchrow("SELECT cumulative_tokens FROM events WHERE id = $1", m2.id)
        drop = drop_row["cumulative_tokens"]

        events = await queries.read_windowed_context_events(
            conn, session_id, account_id=account_id, drop=drop
        )

    seqs = {e.seq for e in events}
    # Retained messages present, dropped messages absent.
    assert m3.seq in seqs and m4.seq in seqs
    assert m1.seq not in seqs and m2.seq not in seqs
    # notice_a (seq <= m2.seq, in the dropped prefix) is excluded.
    assert notice_a.seq not in seqs
    # notice_b (seq > m2.seq, in the retained window) is included.
    assert notice_b.seq in seqs
