"""Integration tests: ``services.append_tool_result`` is idempotent on
``(session_id, tool_call_id)``.

Pre-fix: the custom-tool result intake at
``POST /v1/sessions/{id}/tool-results`` (``api/routers/sessions.py:350``)
and the connector-runtime intake at
``POST /v1/connectors/runtime/tool-results`` (``api/routers/connectors.py:332``)
both forwarded straight to ``services.append_tool_result``, which
unconditionally appended a new tool-role event via ``append_event``. A
network retry (502, transient client disconnect, mid-flight timeout)
appends a *second* tool-role event with the same ``tool_call_id``.

Two consequences:

1. **Monotonic context invariant violated** (CLAUDE.md key invariant #2):
   ``harness/context.py:499-506`` builds ``real_results: dict[tcid →
   data]`` by iterating events and overwriting the dict, so a duplicate
   with different content silently rewrites the model's view of an
   earlier turn — the prompt cache invalidates and history changes
   after the fact.
2. ``cumulative_tokens`` double-counts the duplicate, perturbing the
   windowing-overhead estimate; the duplicate also burns a seq number.

The companion ``mark_management_call_resolved`` (``queries.py``) uses
``WHERE status = 'pending'`` as an atomic dedup. This test pins the same
guarantee for tool-result intake.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_with_parent_tool_call(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, session_id, tool_call_id)`` for an
    initialized session with one assistant event carrying a ``tool_calls``
    entry. ``append_tool_result`` will find a matching parent via
    ``lookup_tool_name_by_call_id``."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_test', NULL, TRUE, 'tenant-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_test", prefix="dedup-test", tools=[ToolSpec(type="bash")]
        )
        async with pool.acquire() as conn:
            # Append the parent assistant event with a tool_calls entry.
            await queries.append_event(
                conn,
                account_id="acc_test",
                session_id=session.id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_dup_test",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
            )
        yield pool, "acc_test", session.id, "tc_dup_test"
    finally:
        await pool.close()


async def _count_tool_results(pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str) -> int:
    """Count tool-role events for this (session_id, tool_call_id)."""
    async with pool.acquire() as conn:
        return (
            await conn.fetchval(
                """
            SELECT COUNT(*) FROM events
             WHERE session_id = $1
               AND kind = 'message'
               AND data->>'role' = 'tool'
               AND data->>'tool_call_id' = $2
            """,
                session_id,
                tool_call_id,
            )
            or 0
        )


class TestAppendToolResultIdempotency:
    async def test_duplicate_post_does_not_create_second_event(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """Two appends with the same ``tool_call_id`` must produce ONE
        event in the session log. Today the second call creates a
        duplicate; this test fails with 2 events instead of 1."""
        pool, account_id, session_id, tool_call_id = session_with_parent_tool_call

        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content="first",
            )
        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content="second-retry",
            )

        count = await _count_tool_results(pool, session_id, tool_call_id)
        assert count == 1, (
            f"duplicate POST appended a second tool_result event "
            f"(count={count}); monotonic-context invariant violated"
        )

    async def test_duplicate_post_returns_original_content(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """The idempotent return must carry the *first* content. Returning
        the duplicate's content would silently corrupt prior history."""
        pool, account_id, session_id, tool_call_id = session_with_parent_tool_call

        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content="first",
            )
        async with pool.acquire() as conn:
            second_event = await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content="second-retry",
            )

        assert second_event.data["content"] == "first", (
            f"duplicate POST returned the second-call's content "
            f"(data={second_event.data!r}); idempotent return must "
            f"preserve the first-call's truth"
        )


async def _open_tool_call_count(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        return (
            await conn.fetchval(
                "SELECT open_tool_call_count FROM sessions WHERE id = $1", session_id
            )
            or 0
        )


class TestAppendToolResultSingleScan:
    """Issue #991 Part 2: ``append_tool_result`` resolves the parent name AND
    channel in ONE ``@>`` scan (``lookup_tool_name_by_call_id`` now projects
    both), then feeds the channel as ``tool_parent_channel`` — so the second
    byte-identical ``_lookup_tool_parent_channel`` scan is never invoked."""

    async def test_parent_channel_lookup_not_invoked(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pool, account_id, session_id, tool_call_id = session_with_parent_tool_call

        from aios.db.queries import events as events_mod

        calls = {"n": 0}
        real = events_mod._lookup_tool_parent_channel

        async def _spy(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return await real(*args, **kwargs)

        # Patch on the facade (callers go through ``queries._lookup_tool_parent_channel``)
        # AND the module (``precompute_event_append`` calls the module-local name).
        monkeypatch.setattr(events_mod, "_lookup_tool_parent_channel", _spy)
        monkeypatch.setattr(queries, "_lookup_tool_parent_channel", _spy)

        async with pool.acquire() as conn:
            ev = await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content="ok",
            )
        # The stored channel still resolves correctly (defaults NULL focal here).
        assert ev.channel is None
        assert calls["n"] == 0, (
            "append_tool_result must not invoke _lookup_tool_parent_channel — "
            "the single lookup_tool_name_by_call_id scan already projected the channel"
        )


class TestAppendToolResultConcurrentDedup:
    """Issue #991 (defense-in-depth): a TRUE concurrent pair of appends for the
    same ``tool_call_id`` — racing through the session ``FOR UPDATE`` — must
    still produce EXACTLY ONE tool-role event and leave ``open_tool_call_count``
    at 0.  The lock-narrowing of #991 moves only the tokenizer pre-compute off
    the lock; the dedup ``find_tool_result_event`` stays INSIDE it."""

    async def test_gather_two_appends_yields_one_event(
        self,
        session_with_parent_tool_call: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        import asyncio

        pool, account_id, session_id, tool_call_id = session_with_parent_tool_call

        async def _append(content: str) -> None:
            async with pool.acquire() as conn:
                await sessions_service.append_tool_result(
                    conn,
                    account_id=account_id,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    content=content,
                )

        await asyncio.gather(_append("first"), _append("second"))

        count = await _count_tool_results(pool, session_id, tool_call_id)
        assert count == 1, f"concurrent appends produced {count} tool events, expected exactly 1"
        # The parent assistant opened one tool call; the single committed result
        # closes it, and the deduped append decrements the id-blind +1 — net 0.
        assert await _open_tool_call_count(pool, session_id) == 0
