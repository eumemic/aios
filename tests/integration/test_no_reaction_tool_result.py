"""Integration suite for the fire-and-forget (``no_reaction``) tool-result wake fix.

The bug
-------
A session runs one agent across connector channels. When the model calls a
message-delivery tool (``signal_send`` / ``signal_react`` and the telegram /
whatsapp equivalents), the connector runtime executes it and POSTs the result
(``{"sent_at_ms": N}``) to ``POST /v1/connectors/runtime/tool-results``. The
intake unconditionally ``defer_wake``\\ s, and the wake gate counts the
``role='tool'`` result as new unreacted stimulus — so the session RE-INFERS to
react to its own delivery confirmation. There is nothing to react to; a
less-disciplined model RE-SENDS the same message (the duplicate-send loop).

The fix
-------
The connector declares the tool fire-and-forget; on a *successful* result the
runner sets ``no_reaction=true`` on the POST. The intake computes
``no_reaction = body.no_reaction AND NOT body.is_error`` and, when true, appends
the result WITH ``data['no_reaction']=true`` (so the model still sees it) but
SKIPS ``defer_wake``. ``append_event`` then treats a ``no_reaction`` tool result
as a NON-stimulus (it does not bump ``last_stimulus_seq``), and the sweep's
``UNREACTED_ROWS_SQL`` excludes the marked row. Both layers agree, so the
session settles instead of re-inferring.

What this suite pins
--------------------
- ``append_tool_result(no_reaction=True)`` stamps ``data['no_reaction']=true``
  and does NOT advance ``last_stimulus_seq`` → the session is NOT a wake
  candidate (loop closed).
- A FAILED send (``is_error=True``) still wakes — the marker is never stamped
  on the error path (the intake AND-gates with ``not is_error``).
- A non-fire-and-forget result (``no_reaction=False``) still wakes.
- An UNMARKED result (historical / not-yet-redeployed connector) still wakes
  exactly as before (backward-compat: missing key → counts as stimulus).
- A co-pending REAL user message still wakes even when a fire-and-forget result
  is also unreacted (no false negative).
- The model still SEES the result in the log (it stays appended).
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
    """Return ``data->>'no_reaction'`` for the tool-result row, or None."""
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
        account_id = "acc_no_reaction"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "no-reaction",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="no-reaction",
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
    no_reaction: bool,
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
    # The connector POSTs the result. ``append_tool_result`` is the shared
    # intake target; ``no_reaction`` is what the API handler computes
    # (``body.no_reaction AND NOT body.is_error``).
    async with pool.acquire() as conn:
        await sessions_service.append_tool_result(
            conn,
            account_id=account_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            content='{"sent_at_ms": 123}',
            is_error=is_error,
            no_reaction=no_reaction,
        )


class TestFireAndForgetSettles:
    async def test_successful_fire_and_forget_does_not_wake(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_account_session
        await _drive_one_send_turn(
            pool, account_id, session_id, tool_call_id="tc_send", no_reaction=True
        )

        # The result is appended WITH the marker — the model still sees it.
        assert await _result_marker(pool, session_id, "tc_send") == "true"

        # The marker keeps the result from counting as a stimulus: the scalar
        # gate (last_stimulus_seq) does not advance past the reacted watermark.
        scalars = await _scalars(pool, session_id)
        assert scalars["last_stimulus_seq"] <= scalars["last_reacted_seq"]
        assert await _status(pool, session_id, account_id) == "idle"

        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id not in needs, (
            "a successful fire-and-forget delivery confirmation must NOT wake the "
            "session — that is the duplicate-send loop"
        )

    async def test_failed_send_still_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_account_session
        # The intake AND-gates ``no_reaction`` with ``not is_error``; a failure
        # arrives with no_reaction already False (the runner's error branches
        # never set it). Pass no_reaction=False to model that.
        await _drive_one_send_turn(
            pool,
            account_id,
            session_id,
            tool_call_id="tc_fail",
            no_reaction=False,
            is_error=True,
        )

        assert await _result_marker(pool, session_id, "tc_fail") is None
        assert await _status(pool, session_id, account_id) == "active"

        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs, "a FAILED send must wake so the model can recover"

    async def test_non_fire_and_forget_result_still_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = pool_account_session
        # A list/get/edit-style tool result carries data the model must consume.
        await _drive_one_send_turn(
            pool, account_id, session_id, tool_call_id="tc_list", no_reaction=False
        )

        assert await _result_marker(pool, session_id, "tc_list") is None
        assert await _status(pool, session_id, account_id) == "active"

        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs

    async def test_unmarked_result_wakes_backward_compat(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A historical event / not-yet-redeployed connector omits the field.
        ``append_event`` sees no ``no_reaction`` key → stimulus → wakes; the
        sweep's ``IS DISTINCT FROM 'true'`` is NULL-safe → row still counts."""
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
                data=_assistant(["tc_old"], reacting_to=1),
            )
            # Append the tool-result event directly with NO no_reaction key —
            # exactly what a pre-fix / historical row looks like.
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": "tc_old",
                    "content": "ok",
                    "name": "signal_send",
                },
            )

        assert await _result_marker(pool, session_id, "tc_old") is None
        assert await _status(pool, session_id, account_id) == "active"

        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs

    async def test_copending_user_message_still_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A real user message landing alongside the fire-and-forget result
        must still wake — the user stimulus is independent of the ack."""
        pool, account_id, session_id = pool_account_session
        await _drive_one_send_turn(
            pool, account_id, session_id, tool_call_id="tc_send", no_reaction=True
        )
        # The fire-and-forget result alone settled the session; now a real user
        # message arrives.
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
    async def test_no_reaction_completing_mixed_batch_still_wakes(
        self,
        pool_account_session: tuple[asyncpg.Pool[Any], str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A turn emits BOTH a real builtin (``bash``) AND a fire-and-forget connector
        send. The real result lands first (unreacted; batch still incomplete), then the
        ``no_reaction`` send result lands LAST via the connector-runtime intake, COMPLETING
        the batch. The session now has an unreacted real result and must be woken — but the
        intake's ``no_reaction`` wake-skip drops the wake, stranding the real result until
        the 30s periodic sweep (~30s of dead air). The wake decision belongs to the gate
        (which excludes the ``no_reaction`` row), not this intake site, which cannot see the
        in-flight real sibling."""
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
                no_reaction=False,
            )

        # The fire-and-forget send result lands LAST, through the connector-runtime
        # intake — the site whose wake decision is under test.
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
                no_reaction=True,
            ),
            pool,
            auth,
        )

        # The gate agrees there is work: the real bash result is unreacted and the
        # batch is now complete (this assertion holds on buggy code too — the gate is
        # correct; the bug is the missing wake to act on it).
        registry = InflightToolRegistry()
        needs = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in needs

        # The intake MUST enqueue a wake. Skipping it on no_reaction stranded the real
        # sibling until the periodic sweep — the bug.
        assert defer_wake_mock.called, (
            "a no_reaction result completing a mixed batch with an unreacted real "
            "sibling must wake the session, not strand it until the 30s periodic sweep"
        )
