"""Sweep behavior for sessions parked in ``errored`` status (#353).

Asserts the two halves of the fix:

- ``errored`` is opaque to ``find_sessions_needing_inference`` even when
  the session has an unreacted user message (the loop trigger #353 hit).
- A fresh user message lifts ``errored → pending`` via ``append_event``
  itself, restoring the operator-recovery contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.sweep import find_sessions_needing_inference
from aios.models.events import ERRORED_LIFECYCLE_STATUS, ERRORED_LIFECYCLE_STOP_REASON
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker

pytestmark = pytest.mark.docker


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
    errored: bool = False,
) -> str:
    """Create a session, append an unreacted user message, and optionally park
    it in the derived ``errored`` state.  Returns the new session id.
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
    if errored:
        # ``errored`` is derived from the event log (sweep.ERRORED_SESSIONS_SQL /
        # queries._SESSION_ERRORED_EXPR): a ``turn_ended``/``error`` lifecycle
        # event more recent than the last user message. Append the same event
        # ``loop._apply_retry_or_failure`` writes so the derived exclusion fires.
        await sessions_service.append_event(
            pool,
            session.id,
            "lifecycle",
            {
                "event": "turn_ended",
                "status": ERRORED_LIFECYCLE_STATUS,
                "stop_reason": ERRORED_LIFECYCLE_STOP_REASON,
            },
            account_id=account_id,
        )
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
        sid = await _seed_session_with_unreacted_message(db_pool, errored=True)

        result = await find_sessions_needing_inference(
            db_pool, InflightToolRegistry(), session_id=sid
        )

        assert result == set(), (
            "Errored sessions must not be re-woken by the sweep — that's the regression #353 fixes."
        )

    async def test_active_session_still_picked_up(self, db_pool: asyncpg.Pool[Any]) -> None:
        """Sanity: a non-errored session with an unreacted user msg is found."""
        sid = await _seed_session_with_unreacted_message(db_pool, errored=False)

        result = await find_sessions_needing_inference(
            db_pool, InflightToolRegistry(), session_id=sid
        )

        assert sid in result

    async def test_user_message_recovers_errored_session(self, db_pool: asyncpg.Pool[Any]) -> None:
        """A new user message is the operator-recovery escape hatch: its seq
        overtakes the error lifecycle event, so the session stops deriving
        ``errored`` and the sweep picks it up again — no status flip needed.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        sid = await _seed_session_with_unreacted_message(db_pool, errored=True)

        # Parked: the sweep skips it.
        assert (
            await find_sessions_needing_inference(db_pool, InflightToolRegistry(), session_id=sid)
            == set()
        )

        await sessions_service.append_user_message(
            db_pool, sid, "please try again", account_id=account_id
        )

        # Recovered: the new user message is more recent than the error event,
        # so the session is no longer derived-errored and the sweep finds it.
        result = await find_sessions_needing_inference(
            db_pool, InflightToolRegistry(), session_id=sid
        )
        assert sid in result, (
            "A new user message must recover an errored session so the next "
            "wake actually runs — otherwise recovery requires manual DB poking."
        )
