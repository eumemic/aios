"""Integration suite for the every-tool-result-is-a-stimulus wake behavior (#1919).

Background
----------
A session runs one agent across connector channels. When the model calls a
message-delivery tool (``signal_send`` / ``signal_react`` and the telegram /
whatsapp equivalents), the connector runtime executes it and POSTs the result
(``{"sent_at_ms": N}``) to ``POST /v1/connectors/runtime/tool-results``. The
intake unconditionally ``defer_wake``\\ s, and the wake gate counts the
``role='tool'`` result as new unreacted stimulus — so the session wakes and the
model gets a turn to react to the completion.

#1398/#1121/#1489 briefly special-cased this: a connector declared the tool
fire-and-forget, the runner POSTed ``no_reaction=true`` on success, and both the
scalar gate (``last_stimulus_seq``) and the sweep excluded the marked row so the
session settled idle. That suppression existed to placate a dumber model that
re-sent on a bare delivery ack. #1919 removed it: **every tool result is a
stimulus.** The loop is the model's responsibility to avoid; the cost of a
spurious "anything else?" wake is cheap, silent idle is not.

What this suite pins
--------------------
- A lone successful ``signal_send`` delivery result IS a stimulus: it advances
  ``last_stimulus_seq``, leaves the session ACTIVE, and the sweep wakes it.
- A FAILED send wakes (as every result does), so the model can recover.
- A non-delivery result (``bash`` etc.) wakes.
- A historical row carrying an inert ``data['no_reaction']=true`` marker (from
  before the removal) now wakes too — ``is_stimulus`` no longer reads the key.
- A fire-and-forget send completing a MIXED batch wakes: the intake always
  ``defer_wake``\\ s and the gate sees the unreacted real sibling.
- The model SEES every result in the log (it stays appended).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.api.routers import connectors as connectors_router
from aios.api.routers.connectors import RuntimeToolResultRequest
from aios.db import queries
from aios.db.pool import create_pool
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.sweep import find_sessions_needing_inference
from aios.models.agents import ToolSpec
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


def _assistant(tool_call_ids: list[str], *, reacting_to: int | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tcid,
                "type": "function",
                "function": {"name": "signal_send", "arguments": "{}"},
            }
            for tcid in tool_call_ids
        ],
    }
    if reacting_to is not None:
        data["reacting_to"] = reacting_to
    return data


async def _scalars(pool: asyncpg.Pool[Any], session_id: str) -> dict[str, int]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT last_stimulus_seq, last_reacted_seq FROM sessions WHERE id = $1",
            session_id,
        )
    assert row is not None
    return dict(row)


async def _status(pool: asyncpg.Pool[Any], session_id: str, account_id: str) -> str:
    async with pool.acquire() as conn:
        session = await queries.get_session(conn, session_id, account_id=account_id)
    return session.status


async def _result_marker(pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str) -> Any:
    """Return ``data->>'no_reaction'`` for the tool-result row, or None.

    ``no_reaction`` is no longer written by any code path; this reads the raw
    JSONB only so the legacy-inert-marker test can prove a historical value is
    present-but-ignored."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT data->>'no_reaction' FROM events "
            "WHERE session_id = $1 AND kind = 'message' AND role = 'tool' "
            "AND data->>'tool_call_id' = $2",
            session_id,
            tool_call_id,
        )


@pytest.fixture
async def pool_account_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """``(pool, account_id, session_id)`` for a fresh session whose agent
    declares a tool the parent ``tool_calls`` entry can resolve a name from."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_tool_stimulus"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "tool-stimulus",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="tool-stimulus",
            tools=[ToolSpec(type="bash")],
        )
        yield pool, account_id, session.id
    finally:
        await pool.close()


async def _drive_one_send_turn(
    pool: asyncpg.Pool[Any],
    account_id: str,
    session_id: str,
    *,
    tool_call_id: str,
    is_error: bool = False,
) -> None:
    """User → assistant(tool_call) → reacted → tool result. Mirrors the live
    flow: the assistant has already reacted to the user, so the only thing
    left in the log after the result is the result itself."""
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "say hi to alice"},
        )
        # Assistant reacts to the user (seq 1) and issues the send tool call.
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data=_assistant([tool_call_id], reacting_to=1),
        )
    # The connector POSTs the result. ``append_tool_result`` is the shared intake.
    async with pool.acquire() as conn:
        await sessions_service.append_tool_result(
            conn,
            account_id=account_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            content='{"sent_at_ms": 123}',
            is_error=is_error,
        )


class TestEveryToolResultWakes:
    async def test_lone_successful_send_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A lone ``signal_send`` delivery ack IS a stimulus (#1919): it advances
        the scalar gate past the reacted watermark, leaves the session ACTIVE,
        and the sweep wakes it so the model gets a turn to react."""
        pool, account_id, session_id = pool_account_session
        await _drive_one_send_turn(
            pool, account_id, session_id, tool_call_id="tc_send"
        )

        # The result is appended (the model sees it) with no suppression marker.
        assert await _result_marker(pool, session_id, "tc_send") is None

        # The result counts as a stimulus: the scalar gate advances past the
        # reacted watermark (seq 1), so there is unreacted work.
        scalars = await _scalars(pool, session_id)
        assert scalars["last_stimulus_seq"] > scalars["last_reacted_seq"]
        assert await _status(pool, session_id, account_id) == "active"

        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs, (
            "a lone signal_send delivery result must wake the session — every "
            "tool result is a stimulus (#1919)"
        )

    async def test_failed_send_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_account_session
        await _drive_one_send_turn(
            pool,
            account_id,
            session_id,
            tool_call_id="tc_fail",
            is_error=True,
        )

        assert await _status(pool, session_id, account_id) == "active"

        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs, "a FAILED send must wake so the model can recover"

    async def test_non_delivery_result_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_account_session
        # A list/get/edit-style tool result carries data the model must consume.
        await _drive_one_send_turn(
            pool, account_id, session_id, tool_call_id="tc_list"
        )

        assert await _status(pool, session_id, account_id) == "active"

        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs

    async def test_legacy_no_reaction_marker_is_inert_and_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A historical row still carrying ``data['no_reaction']=true`` (written
        before #1919) now wakes exactly like any other result: ``is_stimulus``
        no longer reads the key and the sweep no longer excludes the row."""
        pool, account_id, session_id = pool_account_session
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "say hi"},
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=_assistant(["tc_legacy"], reacting_to=1),
            )
            # Append the tool-result event directly WITH a legacy no_reaction
            # marker — exactly what a pre-#1919 row on disk looks like.
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": "tc_legacy",
                    "content": '{"sent_at_ms": 1}',
                    "name": "signal_send",
                    "no_reaction": True,
                },
            )

        # The marker is still on the row (we never rewrite history) …
        assert await _result_marker(pool, session_id, "tc_legacy") == "true"
        # … but it is inert: the row counts as a stimulus and the session wakes.
        assert await _status(pool, session_id, account_id) == "active"

        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs, (
            "a historical no_reaction-marked row must now count as a stimulus — "
            "the marker is inert after #1919"
        )

    async def test_copending_user_message_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A real user message landing alongside a delivery result still wakes."""
        pool, account_id, session_id = pool_account_session
        await _drive_one_send_turn(
            pool, account_id, session_id, tool_call_id="tc_send"
        )
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "actually, also tell bob"},
            )

        assert await _status(pool, session_id, account_id) == "active"
        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs, "a co-pending real user message must still wake"


_CONNECTOR = "signal"


async def _bind_connection(pool: asyncpg.Pool[Any], account_id: str, session_id: str) -> str:
    """Create a ``signal`` connection and bind ``session_id`` to it; return its id —
    the runtime intake requires the session to be bound to ``body.connection_id``."""
    async with pool.acquire() as conn:
        connection = await queries.insert_connection(
            conn,
            account_id=account_id,
            connector=_CONNECTOR,
            external_account_id="+15550000000",
            metadata={},
        )
        await queries.insert_binding(
            conn,
            account_id=account_id,
            connection_id=connection.id,
            mode="single_session",
            session_id=session_id,
        )
    return connection.id


class TestMixedBatchWake:
    async def test_send_completing_mixed_batch_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A turn emits BOTH a real builtin (``bash``) AND a connector send. The
        real result lands first (unreacted; batch still incomplete), then the
        send result lands LAST via the connector-runtime intake, COMPLETING the
        batch. The intake always ``defer_wake``\\ s and the gate sees the
        unreacted real sibling, so the session wakes immediately — never stranded
        until the 30s periodic sweep."""
        pool, account_id, session_id = pool_account_session
        connection_id = await _bind_connection(pool, account_id, session_id)

        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={"role": "user", "content": "run ls and tell alice"},
            )
            # Assistant reacts to user@1 and emits a MIXED batch: a real builtin + a send.
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "reacting_to": 1,
                    "tool_calls": [
                        {
                            "id": "tc_real",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                        {
                            "id": "tc_send",
                            "type": "function",
                            "function": {"name": "signal_send", "arguments": "{}"},
                        },
                    ],
                },
            )
            # The REAL tool result lands first: unreacted, and the batch is still
            # incomplete (tc_send pending), so nothing wakes yet — correct.
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session_id,
                tool_call_id="tc_real",
                content="file1\nfile2",
                is_error=False,
            )

        # The send result lands LAST, through the connector-runtime intake — the
        # site whose wake decision is under test.
        defer_wake_mock = AsyncMock()
        monkeypatch.setattr(connectors_router, "defer_wake", defer_wake_mock)
        auth = ("runtime-token", _CONNECTOR, account_id, None)
        await connectors_router.post_runtime_tool_result(
            RuntimeToolResultRequest(
                connection_id=connection_id,
                session_id=session_id,
                tool_call_id="tc_send",
                content='{"sent_at_ms": 1}',
                is_error=False,
            ),
            pool,
            auth,
        )

        # The gate agrees there is work: the real bash result is unreacted and the
        # batch is now complete.
        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs

        # The intake always enqueues a wake now — nothing is stranded.
        assert defer_wake_mock.called, (
            "a result completing a mixed batch with an unreacted real sibling must "
            "wake the session, not strand it until the 30s periodic sweep"
        )
