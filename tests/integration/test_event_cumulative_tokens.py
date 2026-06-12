"""Integration test: ``append_event`` keeps ``cumulative_tokens`` a running
sum even after the tokenizer pass moved BEFORE the row lock (issue #862).

The per-event token delta is now computed pre-transaction; the running sum
(``prev + delta``, prev fetched under the lock) must be byte-identical to the
old in-lock computation. We append a scripted sequence and assert each stored
``cumulative_tokens`` equals the expected partial sum computed independently
via ``_event_token_delta``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries.events import _event_token_delta
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
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_cumtok', NULL, TRUE, 'cumtok-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_cumtok", prefix="cumtok-test"
        )
        yield pool, "acc_cumtok", session.id
    finally:
        await pool.close()


async def _cumulative(
    conn: asyncpg.Connection[Any], session_id: str
) -> list[tuple[str, int | None]]:
    """Return ``[(role, cumulative_tokens), ...]`` in seq order for messages."""
    rows = await conn.fetch(
        "SELECT data->>'role' AS role, cumulative_tokens FROM events "
        "WHERE session_id = $1 AND kind = 'message' ORDER BY seq",
        session_id,
    )
    return [(r["role"], r["cumulative_tokens"]) for r in rows]


async def test_cumulative_tokens_running_sum_unchanged(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, session_id = pool_and_session

    # A scripted conversation: user, assistant-with-tool_calls, tool, user.
    scripted: list[dict[str, Any]] = [
        {"role": "user", "content": "what files are here"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tc_1", "name": "bash", "content": "a.txt\nb.txt"},
        {"role": "user", "content": "thanks, now read a.txt"},
    ]

    async with pool.acquire() as conn:
        for data in scripted:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=data,
                orig_channel=None,
            )

        stored = await _cumulative(conn, session_id)

    # Expected running sum, computed independently via the pure delta helper.
    # focal is None throughout (no switch_channel), orig_channel None.
    expected: list[int] = []
    running = 0
    for data in scripted:
        running += _event_token_delta("message", data, None, None)
        expected.append(running)

    assert [c for _role, c in stored] == expected
    # Sanity: every value is a strictly increasing positive running sum.
    assert all(c is not None and c > 0 for _role, c in stored)
    assert expected == sorted(expected)
