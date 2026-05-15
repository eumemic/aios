"""Sweep behavior for sessions parked in ``errored`` status (#353).

Asserts the two halves of the fix:

- ``errored`` is opaque to ``find_sessions_needing_inference`` even when
  the session has an unreacted user message (the loop trigger #353 hit).
- A fresh user message lifts ``errored → pending`` via
  ``flip_quiescent_to_pending``, restoring the operator-recovery contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.harness.sweep import find_sessions_needing_inference
from aios.harness.task_registry import TaskRegistry
from aios.models.sessions import SessionStatus
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker


async def _ensure_agent_and_env(pool: asyncpg.Pool[Any]) -> tuple[str, str]:
    """Idempotently seed the FK targets ``create_session`` needs."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agents (id, name, model, account_id) "
            "VALUES ('agt_err', 'err', 'openrouter/x', 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING"
        )
        await conn.execute(
            "INSERT INTO environments (id, name, account_id) "
            "VALUES ('env_err', 'env_err', 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING"
        )
    return "agt_err", "env_err"


async def _seed_session_with_unreacted_message(
    pool: asyncpg.Pool[Any],
    *,
    status: SessionStatus,
) -> str:
    """Create a session via the service layer, append a user message, then
    force ``status`` if it isn't ``pending``.  Returns the new session id.
    """
    account_id = "acc_test_stub"  # PR 3 scaffolding
    agent_id, environment_id = await _ensure_agent_and_env(pool)
    session = await sessions_service.create_session(
        pool,
        agent_id=agent_id,
        environment_id=environment_id,
        title=None,
        metadata={},
        account_id=account_id,
    )
    await sessions_service.append_user_message(pool, session.id, "hi", account_id=account_id)
    if status != "pending":
        await sessions_service.set_session_status(pool, session.id, status, account_id=account_id)
    return session.id


@pytest.fixture
async def db_pool(aios_env: dict[str, str]) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(aios_env["AIOS_DB_URL"], min_size=1, max_size=2)
    try:
        yield pool
    finally:
        await pool.close()


@needs_docker
class TestErroredSessionSweepFilter:
    async def test_errored_session_excluded_from_find_sessions_needing_inference(
        self, db_pool: asyncpg.Pool[Any]
    ) -> None:
        """The exhaustion landing pad: errored + unreacted user msg ⇒ skipped."""
        sid = await _seed_session_with_unreacted_message(db_pool, status="errored")

        result = await find_sessions_needing_inference(db_pool, TaskRegistry(), session_id=sid)

        assert result == set(), (
            "Errored sessions must not be re-woken by the sweep — that's the regression #353 fixes."
        )

    async def test_pending_session_still_picked_up(self, db_pool: asyncpg.Pool[Any]) -> None:
        """Sanity: a pending session with an unreacted user msg is still found."""
        sid = await _seed_session_with_unreacted_message(db_pool, status="pending")

        result = await find_sessions_needing_inference(db_pool, TaskRegistry(), session_id=sid)

        assert sid in result

    async def test_user_message_flips_errored_to_pending(self, db_pool: asyncpg.Pool[Any]) -> None:
        """``append_user_message`` is the operator-recovery escape hatch:
        it must un-park an ``errored`` session by flipping status to ``pending``.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        sid = await _seed_session_with_unreacted_message(db_pool, status="errored")

        await sessions_service.append_user_message(
            db_pool, sid, "please try again", account_id=account_id
        )

        async with db_pool.acquire() as conn:
            status = await conn.fetchval("SELECT status FROM sessions WHERE id = $1", sid)
        assert status == "pending", (
            "A new user message must clear the errored marker so the next "
            "wake actually runs — otherwise recovery requires manual DB poking."
        )
