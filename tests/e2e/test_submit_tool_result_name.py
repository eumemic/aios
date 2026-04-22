"""E2E regression test for issue #133.

When the API receives a custom tool result via
``POST /v1/sessions/{id}/tool-results``, the appended tool-role event must
carry the tool's ``name`` so the derived ``tool_name`` column on the
``events`` table (see migration 0022 + ``_derive_tool_name`` in
``db/queries.py``) is populated.  Without it, ``events_search`` queries
filtering ``WHERE tool_name = '<custom>'`` silently skip custom results.

The server knows the name — it lives on the parent assistant's
``data->'tool_calls'`` — so the client should not have to pass it.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest


def _uniq() -> str:
    return secrets.token_hex(4)


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    settings = get_settings()
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    transport = httpx.ASGITransport(app=app)
    # No real worker in these tests; avoid enqueueing a wake for the
    # procrastinate MagicMock.
    with mock.patch(
        "aios.api.routers.sessions.defer_wake",
        new_callable=mock.AsyncMock,
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {aios_env['AIOS_API_KEY']}"},
        ) as client:
            yield client


@pytest.fixture
async def session_id(pool: Any) -> str:
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(conn, name=f"custom-tool-env-{_uniq()}")
    agent = await agents_svc.create_agent(
        pool,
        name=f"custom-tool-agent-{_uniq()}",
        model="openai/gpt-4o-mini",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    session = await sessions_svc.create_session(
        pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
    )
    return session.id


async def _seed_assistant_tool_call(
    pool: Any,
    session_id: str,
    *,
    tool_call_id: str,
    tool_name: str,
) -> None:
    """Append an assistant event that requests a single custom tool call."""
    from aios.services import sessions as sessions_svc

    await sessions_svc.append_user_message(pool, session_id, "what is the weather?")
    await sessions_svc.append_event(
        pool,
        session_id,
        "message",
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"city": "SF"}),
                    },
                }
            ],
        },
    )


class TestSubmitToolResultStampsName:
    """Regression coverage for issue #133."""

    async def test_stamps_tool_name_from_parent_assistant(
        self,
        http_client: httpx.AsyncClient,
        pool: Any,
        session_id: str,
    ) -> None:
        """Submitting a result matches the parent's tool_call name into the event."""
        call_id = f"call_{_uniq()}"
        await _seed_assistant_tool_call(
            pool,
            session_id,
            tool_call_id=call_id,
            tool_name="get_weather",
        )

        r = await http_client.post(
            f"/v1/sessions/{session_id}/tool-results",
            json={"tool_call_id": call_id, "content": '{"temp_f": 72}'},
        )
        assert r.status_code == 201, r.text
        event = r.json()

        assert event["kind"] == "message"
        assert event["data"]["role"] == "tool"
        assert event["data"]["tool_call_id"] == call_id
        # The regression: the server should promote the parent's tool_call
        # function name into the appended event's data.
        assert event["data"].get("name") == "get_weather"

    async def test_derived_tool_name_column_populated(
        self,
        http_client: httpx.AsyncClient,
        pool: Any,
        session_id: str,
    ) -> None:
        """The physical ``tool_name`` column on events picks up the name.

        A custom tool result queried through ``events_search`` / the raw
        ``events`` table should be discoverable by ``tool_name`` the same
        way built-in and MCP tool results are.
        """
        call_id = f"call_{_uniq()}"
        await _seed_assistant_tool_call(
            pool,
            session_id,
            tool_call_id=call_id,
            tool_name="get_weather",
        )

        r = await http_client.post(
            f"/v1/sessions/{session_id}/tool-results",
            json={"tool_call_id": call_id, "content": "ok"},
        )
        assert r.status_code == 201, r.text
        new_seq = r.json()["seq"]

        async with pool.acquire() as conn:
            tool_name = await conn.fetchval(
                "SELECT tool_name FROM events WHERE session_id = $1 AND seq = $2",
                session_id,
                new_seq,
            )
        assert tool_name == "get_weather"

    async def test_unknown_call_id_returns_404(
        self,
        http_client: httpx.AsyncClient,
        pool: Any,
        session_id: str,
    ) -> None:
        """Submitting a result for a ``tool_call_id`` with no parent is a
        client bug; the server rejects it and appends no event.

        An orphan tool-role event (no matching assistant tool_call) would
        leave a row with NULL ``tool_name`` in ``events`` that cannot be
        reconciled with any parent.  Per CLAUDE.md's "fail hard, no
        fallbacks" rule, the server refuses to store it.
        """
        r = await http_client.post(
            f"/v1/sessions/{session_id}/tool-results",
            json={"tool_call_id": "call_nonexistent", "content": "whatever"},
        )
        assert r.status_code == 404, r.text
        assert r.json()["error"]["type"] == "not_found"

        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM events "
                "WHERE session_id = $1 AND kind = 'message' AND data->>'role' = 'tool'",
                session_id,
            )
        assert count == 0
