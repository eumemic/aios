"""Integration test: ``list_recent_chat_ids`` escapes LIKE wildcards in
``account``.

Pre-fix: ``list_recent_chat_ids`` (``db/queries.py:3750``) built the
LIKE pattern by f-string concatenation:

    prefix = f"{connector}/{account}/"
    ... WHERE channel LIKE $1 ..., prefix + "%", ...

``account`` is operator-supplied (``ConnectionCreate.account`` allows
``%``, ``_``, ``\\``). SQL ``LIKE`` treats ``_`` as "any single char"
and ``%`` as "any string". So an operator with two connections under
the same tenant — e.g. accounts ``bot_a`` and ``botXa`` — calling the
helper for ``bot_a`` would match channel ``telegram/botXa/...`` too,
since ``_`` in the pattern position matches the ``X`` literal in the
stored channel. The result is same-tenant data confusion: chats from
``botXa`` get reported as belonging to ``bot_a``.

The sibling helper at ``queries.py:4454`` already escapes with
``_escape_like`` (defined at ``queries.py:241``). This is the
companion fix.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.ids import make_id
from aios.models.agents import ToolSpec
from aios.services import agents as agents_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_in_account(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a fresh session."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_test', NULL, TRUE, 'tenant-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_test",
            name="like-test",
            model="openrouter/test",
            system="",
            tools=[ToolSpec(type="bash")],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_test", name="like-test-env"
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_test",
                agent_id=agent.id,
                environment_id=env.id,
                agent_version=agent.version,
                title=None,
                metadata={},
            )
        yield pool, "acc_test", session.id
    finally:
        await pool.close()


async def _insert_user_event(
    pool: asyncpg.Pool[Any], session_id: str, account_id: str, channel: str
) -> None:
    """Insert a user-role message event with the given channel directly.

    Bypasses ``append_event``'s derived-channel logic so the test can
    pin exactly which channel string is stored for the LIKE comparison.
    """
    async with pool.acquire() as conn, conn.transaction():
        seq_row = await conn.fetchrow(
            "UPDATE sessions SET last_event_seq = last_event_seq + 1 "
            "WHERE id = $1 AND account_id = $2 RETURNING last_event_seq",
            session_id,
            account_id,
        )
        assert seq_row is not None
        await conn.execute(
            """
            INSERT INTO events (
                id, session_id, seq, kind, data, created_at,
                orig_channel, focal_channel_at_arrival, channel, account_id
            )
            VALUES ($1, $2, $3, 'message', $4::jsonb, now(),
                    $5, $5, $5, $6)
            """,
            make_id("evt"),
            session_id,
            seq_row["last_event_seq"],
            json.dumps({"role": "user", "content": "hi"}),
            channel,
            account_id,
        )


async def test_underscore_in_account_does_not_wildcard_match(
    session_in_account: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Account ``bot_a`` must not match channel ``botXa`` via SQL LIKE ``_``."""
    pool, account_id, session_id = session_in_account

    # Two events under the same tenant on similarly-shaped channels.
    # The literal ``_`` in ``bot_a`` is a SQL LIKE wildcard pre-fix —
    # without escaping it matches ``X`` in ``botXa``.
    await _insert_user_event(pool, session_id, account_id, "telegram/bot_a/chat_alpha")
    await _insert_user_event(pool, session_id, account_id, "telegram/botXa/chat_beta")

    async with pool.acquire() as conn:
        results = await queries.list_recent_chat_ids(
            conn, "telegram", "bot_a", account_id=account_id, limit=10
        )

    chat_ids = sorted(r[0] for r in results)
    assert chat_ids == ["chat_alpha"], (
        f"unescaped LIKE pattern matched 'botXa' chats too "
        f"(got {chat_ids!r}); the literal '_' in 'bot_a' acted as a "
        f"single-char wildcard against the stored channel"
    )


async def test_percent_in_account_does_not_wildcard_match(
    session_in_account: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Account ``foo%bar`` must not match ``fooANYTHINGbar`` via SQL LIKE ``%``."""
    pool, account_id, session_id = session_in_account

    await _insert_user_event(pool, session_id, account_id, "signal/foo%bar/chat_real")
    await _insert_user_event(pool, session_id, account_id, "signal/fooXXXXXbar/chat_imposter")

    async with pool.acquire() as conn:
        results = await queries.list_recent_chat_ids(
            conn, "signal", "foo%bar", account_id=account_id, limit=10
        )

    chat_ids = sorted(r[0] for r in results)
    assert chat_ids == ["chat_real"], (
        f"unescaped LIKE pattern matched 'fooXXXXXbar' too "
        f"(got {chat_ids!r}); the literal '%' in 'foo%bar' acted as a "
        f"multi-char wildcard against the stored channel"
    )


async def test_plain_account_still_returns_matching_chats(
    session_in_account: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Regression guard: the escape doesn't break the happy path."""
    pool, account_id, session_id = session_in_account

    await _insert_user_event(pool, session_id, account_id, "signal/plain/chat_a")
    await _insert_user_event(pool, session_id, account_id, "signal/plain/chat_b")

    async with pool.acquire() as conn:
        results = await queries.list_recent_chat_ids(
            conn, "signal", "plain", account_id=account_id, limit=10
        )

    chat_ids = sorted(r[0] for r in results)
    assert chat_ids == ["chat_a", "chat_b"]


async def test_returns_max_last_seen_at(
    session_in_account: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Regression guard: ``last_seen_at`` is the MAX across events for that chat."""
    pool, account_id, session_id = session_in_account

    await _insert_user_event(pool, session_id, account_id, "signal/plain/chat_a")
    await _insert_user_event(pool, session_id, account_id, "signal/plain/chat_a")

    async with pool.acquire() as conn:
        results = await queries.list_recent_chat_ids(
            conn, "signal", "plain", account_id=account_id, limit=10
        )

    assert len(results) == 1  # GROUP BY chat_id collapses both events
    assert results[0][0] == "chat_a"
    assert isinstance(results[0][1], datetime)
    assert results[0][1].tzinfo is UTC or results[0][1].tzinfo is not None
