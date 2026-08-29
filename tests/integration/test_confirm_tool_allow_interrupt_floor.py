"""Integration test: ``confirm_tool_allow``'s interrupt-floor guard for the
auto-review checker (jarbot#229).

A checker ALLOW is not human intent. The checker captures the latest interrupt
seq before grading; without an in-transaction re-check, an interrupt landing
between the capture and the confirm append yields a ``tool_confirmed`` whose
seq is HIGHER than the interrupt, which the #1756 cold-dispatch guard reads as
fresh human re-confirmation and runs the interrupted call.

``enforce_interrupt_floor=True`` re-checks the captured floor INSIDE the
session-locked transaction: if an interrupt has landed since (floor moved), the
confirm raises :class:`ConflictError` and the checker drops the allow. Humans
never pass the flag — a person clicking Allow after an interrupt IS fresh intent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_with_open_call(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """A session with one unresolved assistant tool_call — no result, no
    confirm. Yields ``(pool, account_id, session_id, tool_call_id)``."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_auto_review_floor"
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ($1, NULL, TRUE, $2)
                """,
                account_id,
                "auto-review-floor-test",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="auto-review-floor",
            tools=[ToolSpec(type="bash")],
        )
        tool_call_id = "tc_floor_1"
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
            )
        yield pool, account_id, session.id, tool_call_id
    finally:
        await pool.close()


async def _append_interrupt(pool: asyncpg.Pool[Any], account_id: str, session_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="interrupt",
            data={"event": "interrupted"},
        )


async def test_allow_confirms_when_floor_unchanged(
    session_with_open_call: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """No interrupt since capture (both None): the auto-review confirm lands."""
    pool, account_id, session_id, tool_call_id = session_with_open_call
    floor = await sessions_service.find_latest_interrupt_seq(
        pool, session_id, account_id=account_id
    )
    assert floor is None
    event = await sessions_service.confirm_tool_allow(
        pool,
        session_id,
        tool_call_id,
        account_id=account_id,
        source="auto_review",
        enforce_interrupt_floor=True,
        expected_interrupt_floor=floor,
    )
    assert event.data == {
        "event": "tool_confirmed",
        "tool_call_id": tool_call_id,
        "result": "allow",
        "source": "auto_review",
    }


async def test_allow_rejected_when_interrupt_landed_since_capture(
    session_with_open_call: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """An interrupt committed after the floor was captured → ConflictError,
    and NO tool_confirmed event is written (the call parks for the sweep)."""
    pool, account_id, session_id, tool_call_id = session_with_open_call
    floor = await sessions_service.find_latest_interrupt_seq(
        pool, session_id, account_id=account_id
    )
    # The user interrupts while the checker is grading.
    await _append_interrupt(pool, account_id, session_id)

    with pytest.raises(ConflictError):
        await sessions_service.confirm_tool_allow(
            pool,
            session_id,
            tool_call_id,
            account_id=account_id,
            source="auto_review",
            enforce_interrupt_floor=True,
            expected_interrupt_floor=floor,
        )
    # No confirm was appended — the interrupted call did not auto-execute.
    async with pool.acquire() as conn:
        existing = await queries.find_tool_confirmed_event(
            conn, session_id, tool_call_id, account_id=account_id
        )
    assert existing is None


async def test_human_allow_ignores_floor_after_interrupt(
    session_with_open_call: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A HUMAN clicking Allow after an interrupt IS fresh intent — no flag, so
    the confirm lands even with an interrupt on the log."""
    pool, account_id, session_id, tool_call_id = session_with_open_call
    await _append_interrupt(pool, account_id, session_id)
    event = await sessions_service.confirm_tool_allow(
        pool,
        session_id,
        tool_call_id,
        account_id=account_id,
    )
    assert event.data["result"] == "allow"
    assert "source" not in event.data
