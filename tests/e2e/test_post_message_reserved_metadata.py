"""``POST /v1/sessions/{id}/messages`` must refuse the reminder metadata key.

``metadata[REMINDER_METADATA_KEY]`` marks harness-authored durable reminder
rows, which ``append_event`` treats as non-stimulus. A client-minted one would
be a user message that never wakes the session — so ``append_user_message``
(the one writer behind every externally-sourced user message) rejects it and
the API answers 422 rather than appending it.
"""

from __future__ import annotations

import secrets
from typing import Any

import httpx
import pytest

from aios.models.events import REMINDER_METADATA_KEY


@pytest.fixture
async def session_id(pool: Any) -> str:
    account_id = "acc_test_stub"
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    tag = secrets.token_hex(4)
    async with pool.acquire() as conn:
        env = await queries.insert_environment(conn, name=f"rm-env-{tag}", account_id=account_id)
    agent = await agents_svc.create_agent(
        pool,
        name=f"rm-agent-{tag}",
        model="fake/test",
        system="You are a test assistant.",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        account_id=account_id,
    )
    session = await sessions_svc.create_session(
        pool,
        agent_id=agent.id,
        environment_id=env.id,
        title=None,
        metadata={},
        account_id=account_id,
    )
    return str(session.id)


class TestReservedReminderMetadata:
    async def test_post_message_with_reminder_key_is_422(
        self, http_client: httpx.AsyncClient, session_id: str, pool: Any
    ) -> None:
        resp = await http_client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "content": "sneaky",
                "metadata": {REMINDER_METADATA_KEY: {"section": "concise"}},
            },
        )
        assert resp.status_code == 422, resp.text
        assert REMINDER_METADATA_KEY in resp.text
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'message'",
                session_id,
            )
        assert count == 0, "a rejected message must not be appended"

    async def test_ordinary_metadata_still_accepted(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        resp = await http_client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"content": "hello", "metadata": {"note": "fine"}},
        )
        assert resp.status_code == 201, resp.text
