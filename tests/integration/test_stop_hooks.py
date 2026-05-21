"""Integration tests for the pluggable Stop hooks primitive (issue #374).

Three DB-level guarantees are exercised here:

1. ``update_session_stop_hook`` round-trips every hook variant (the
   discriminated union must serialise to JSONB and parse back to the
   original concrete model).
2. ``clone_session`` mirrors the parent's ``stop_hook`` — the clone's
   first wake has to inherit the same stop semantics as the parent
   had at clone time.
3. ``update_session_stop_hook`` refuses archived sessions, matching
   the existing ``update_session`` archived-rejection pattern. An UPDATE
   that silently no-ops would leave the caller with a 200 response and
   no behavioural change — exactly the silent-failure shape CLAUDE.md
   forbids.

Harness-level behaviour (supersession, continuation, system-prompt
augmentation) is covered by ``tests/unit/test_stop_hooks.py`` — those
helpers are pure-Python and don't need a DB.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.models.sessions import (
    AlwaysContinueStopHook,
    SelfCheckStopHook,
    StopHookSpec,
    TaskCallStopHook,
)
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a fresh idle session.

    Each test gets its own pool so concurrent runs don't interfere; the
    DB is reset between tests via ``_reset_db_state``.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_stop_hook', NULL, TRUE, 'stop-hook-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_stop_hook", prefix="stop-hook"
        )
        yield pool, "acc_stop_hook", session.id
    finally:
        await pool.close()


class TestUpdateSessionStopHookRoundTrip:
    """Every concrete hook variant must serialise → JSONB → parse back."""

    async def test_self_check_round_trip(
        self, session_pool: tuple[asyncpg.Pool[Any], str, str]
    ) -> None:
        pool, account_id, session_id = session_pool
        hook = SelfCheckStopHook(
            prompt="Have you produced the final answer?",
            stop_on="DONE",
            continuation_message="[Refine your answer]",
        )

        await sessions_service.set_session_stop_hook(
            pool, session_id, account_id=account_id, stop_hook=hook
        )

        # Read back via the service so the full enrichment path runs;
        # the parsed type discriminator confirms the JSONB column round-
        # tripped through ``_row_to_session``.
        got = await sessions_service.get_session(pool, session_id, account_id=account_id)
        assert isinstance(got.stop_hook, SelfCheckStopHook)
        assert got.stop_hook.prompt == "Have you produced the final answer?"
        assert got.stop_hook.stop_on == "DONE"
        assert got.stop_hook.continuation_message == "[Refine your answer]"

    async def test_task_call_round_trip(
        self, session_pool: tuple[asyncpg.Pool[Any], str, str]
    ) -> None:
        pool, account_id, session_id = session_pool

        await sessions_service.set_session_stop_hook(
            pool, session_id, account_id=account_id, stop_hook=TaskCallStopHook()
        )
        got = await sessions_service.get_session(pool, session_id, account_id=account_id)
        assert isinstance(got.stop_hook, TaskCallStopHook)
        # The literal-typed ``tool_name`` defaults; the discriminated
        # union must preserve it through the JSONB round-trip.
        assert got.stop_hook.tool_name == "task_complete"

    async def test_always_continue_round_trip(
        self, session_pool: tuple[asyncpg.Pool[Any], str, str]
    ) -> None:
        pool, account_id, session_id = session_pool
        hook = AlwaysContinueStopHook(continuation_message="[Stay on task]")

        await sessions_service.set_session_stop_hook(
            pool, session_id, account_id=account_id, stop_hook=hook
        )
        got = await sessions_service.get_session(pool, session_id, account_id=account_id)
        assert isinstance(got.stop_hook, AlwaysContinueStopHook)
        assert got.stop_hook.continuation_message == "[Stay on task]"

    async def test_clear_to_none(self, session_pool: tuple[asyncpg.Pool[Any], str, str]) -> None:
        """Setting then clearing returns the column to NULL — the harness's
        no-tools branch then takes the unconditional end_turn path."""
        pool, account_id, session_id = session_pool

        await sessions_service.set_session_stop_hook(
            pool, session_id, account_id=account_id, stop_hook=TaskCallStopHook()
        )
        await sessions_service.set_session_stop_hook(
            pool, session_id, account_id=account_id, stop_hook=None
        )
        got = await sessions_service.get_session(pool, session_id, account_id=account_id)
        assert got.stop_hook is None

    async def test_default_is_none(self, session_pool: tuple[asyncpg.Pool[Any], str, str]) -> None:
        """Fresh sessions have no stop_hook — preserves backwards-compatible
        ``end_turn`` semantics for callers that never opt in."""
        pool, account_id, session_id = session_pool
        got = await sessions_service.get_session(pool, session_id, account_id=account_id)
        assert got.stop_hook is None


class TestUpdateSessionStopHookArchived:
    """Archived sessions must reject stop_hook writes (no silent no-op)."""

    async def test_rejects_archived_session(
        self, session_pool: tuple[asyncpg.Pool[Any], str, str]
    ) -> None:
        pool, account_id, session_id = session_pool

        async with pool.acquire() as conn:
            await queries.archive_session(conn, session_id, account_id=account_id)

        with pytest.raises(ConflictError):
            await sessions_service.set_session_stop_hook(
                pool, session_id, account_id=account_id, stop_hook=TaskCallStopHook()
            )


class TestCloneSessionPreservesStopHook:
    """The clone path's INSERT…SELECT must include the ``stop_hook`` column.

    Without this, a session's clone would silently drop into ``end_turn``
    semantics — a continuous-mode loop that loses its hook on resume.
    """

    @pytest.mark.parametrize(
        "hook",
        [
            SelfCheckStopHook(prompt="Done yet?", stop_on="YES"),
            TaskCallStopHook(),
            AlwaysContinueStopHook(continuation_message="[Keep going]"),
        ],
        ids=["self_check", "task_call", "always_continue"],
    )
    async def test_clone_mirrors_stop_hook(
        self,
        session_pool: tuple[asyncpg.Pool[Any], str, str],
        hook: StopHookSpec,
    ) -> None:
        pool, account_id, session_id = session_pool

        await sessions_service.set_session_stop_hook(
            pool, session_id, account_id=account_id, stop_hook=hook
        )
        clone = await sessions_service.clone_session(pool, session_id, account_id=account_id)

        assert clone.stop_hook is not None
        # Equality on the union via model_dump — type+all fields must match
        # for the JSONB round-trip on the clone side to be considered intact.
        assert clone.stop_hook.model_dump() == hook.model_dump()

    async def test_clone_of_null_hook_is_still_null(
        self, session_pool: tuple[asyncpg.Pool[Any], str, str]
    ) -> None:
        """Default-case regression: a parent with no hook clones to a child
        with no hook (not "" or {})."""
        pool, account_id, session_id = session_pool
        clone = await sessions_service.clone_session(pool, session_id, account_id=account_id)
        assert clone.stop_hook is None
