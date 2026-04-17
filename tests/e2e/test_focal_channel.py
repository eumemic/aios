"""E2E tests for the focal-channel attention model (issue #29 redesign).

Covers event stamping (``orig_channel`` + ``focal_channel_at_arrival``)
at append time, the ``switch_channel`` built-in tool, and (in later
slices) MCP ``_meta`` injection + paradigm/tail blocks.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import pytest

from aios.models.routing_rules import SessionParams


def _uniq() -> str:
    return secrets.token_hex(4)


# ─── shared fixtures (mirrors the test_routing.py pattern) ──────────────────


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
async def env_id(pool: Any) -> str:
    from aios.db import queries

    async with pool.acquire() as conn:
        env = await queries.insert_environment(conn, name=f"focal-test-{_uniq()}")
    return env.id


@pytest.fixture
async def agent_id(pool: Any) -> str:
    from aios.services import agents as svc

    a = await svc.create_agent(
        pool,
        name=f"focal-agent-{_uniq()}",
        model="openai/gpt-4o-mini",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    return a.id


@pytest.fixture
async def vault_id(pool: Any) -> str:
    from aios.services import vaults as svc

    v = await svc.create_vault(pool, display_name="focal-vault", metadata={})
    return v.id


# ─── helpers ────────────────────────────────────────────────────────────────


async def _set_focal(pool: Any, session_id: str, focal: str | None) -> None:
    """Direct DB update until the switch_channel tool (slice 5) lands."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET focal_channel = $1 WHERE id = $2",
            focal,
            session_id,
        )


async def _setup_inbound(pool: Any, agent_id: str, env_id: str, vault_id: str) -> tuple[str, str]:
    """Create a connection + routing rule, return (connection_id, prefix)."""
    from aios.services import channels as ch_svc
    from aios.services import connections as conn_svc

    account = f"focal-{_uniq()}"
    connection = await conn_svc.create_connection(
        pool,
        connector="signal",
        account=account,
        mcp_url="https://m",
        vault_id=vault_id,
        metadata={},
    )
    await ch_svc.create_routing_rule(
        pool,
        prefix=f"signal/{account}",
        target=f"agent:{agent_id}",
        session_params=SessionParams(environment_id=env_id),
    )
    return connection.id, f"signal/{account}"


async def _post_inbound(
    pool: Any, prefix: str, path: str, content: str = "hi"
) -> tuple[str, str, str]:
    """Resolve + append a user inbound message.

    Returns ``(session_id, event_id, address)``. Bypasses the HTTP layer
    and defer_wake so the test is DB-focused.
    """
    from aios.services import channels as ch_svc
    from aios.services import sessions as sess_svc

    address = f"{prefix}/{path}"
    with mock.patch("aios.harness.wake.defer_wake"):
        resolution = await ch_svc.resolve_channel(pool, address)
        event = await sess_svc.append_user_message(
            pool,
            resolution.session_id,
            content,
            metadata={"channel": address},
        )
    return resolution.session_id, event.id, address


# ─── slice 2: stamping tests ────────────────────────────────────────────────


class TestEventStampingFromInbound:
    async def test_inbound_stamps_orig_channel(
        self, pool: Any, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        from aios.db import queries

        _connection_id, prefix = await _setup_inbound(pool, agent_id, env_id, vault_id)
        session_id, _event_id, address = await _post_inbound(pool, prefix, "chat-1")

        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, session_id)

        msg = events[-1]
        assert msg.orig_channel == address

    async def test_inbound_stamps_focal_at_arrival_null_when_phone_down(
        self, pool: Any, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        from aios.db import queries

        _connection_id, prefix = await _setup_inbound(pool, agent_id, env_id, vault_id)
        session_id, _event_id, _address = await _post_inbound(pool, prefix, "chat-1")

        # Session was auto-created with focal_channel NULL (phone down).
        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, session_id)
        msg = events[-1]
        assert msg.focal_channel_at_arrival is None

    async def test_inbound_stamps_focal_at_arrival_when_focused(
        self, pool: Any, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        from aios.db import queries

        _connection_id, prefix = await _setup_inbound(pool, agent_id, env_id, vault_id)
        # First message auto-creates the session.
        session_id, _e1, address = await _post_inbound(pool, prefix, "chat-1")
        # Focus the agent on this channel.
        await _set_focal(pool, session_id, address)

        # Second inbound on the same channel — focal matches orig.
        _sid, _e2, _addr = await _post_inbound(pool, prefix, "chat-1", content="still here")

        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, session_id)
        msg = events[-1]
        assert msg.orig_channel == address
        assert msg.focal_channel_at_arrival == address

    async def test_inbound_stamps_focal_when_orig_differs(
        self, pool: Any, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        """Focal=A, inbound on B → stamp orig=B, focal_at_arrival=A."""
        from aios.db import queries
        from aios.services import channels as ch_svc

        _connection_id, prefix = await _setup_inbound(pool, agent_id, env_id, vault_id)
        # First message on chat-A auto-creates the session.
        session_a_id, _e1, address_a = await _post_inbound(pool, prefix, "chat-A")
        # Focus on A.
        await _set_focal(pool, session_a_id, address_a)
        # Bind chat-B to the same session so a POST to B doesn't create a new session.
        address_b = f"{prefix}/chat-B"
        await ch_svc.create_binding(pool, address=address_b, session_id=session_a_id)
        # Inbound on B while focal is A.
        _sid_b, _e2, _addr_b = await _post_inbound(pool, prefix, "chat-B", content="hi from B")

        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, session_a_id)
        msg = events[-1]
        assert msg.orig_channel == address_b
        assert msg.focal_channel_at_arrival == address_a

    async def test_stamping_reflects_latest_focal_across_serial_appends(
        self,
        pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        """append_event reads focal inside the same txn that allocates seq.

        We can't cleanly race asyncio transactions without adding test-
        internal plumbing, but we can verify that the focal column's
        value at append time is the value stamped on the event, across
        multiple serial updates.  Because the UPDATE used for seq
        allocation also reads focal_channel (RETURNING clause), the read
        is serialized against any concurrent focal mutation.
        """
        from aios.db import queries

        _connection_id, prefix = await _setup_inbound(pool, agent_id, env_id, vault_id)
        session_id, _e1, address_a = await _post_inbound(pool, prefix, "chat-A")

        # Flip focal repeatedly, appending between each flip, and assert
        # each event carries the focal that was in effect at its append.
        expected_stamps: list[str | None] = []
        for focal in (None, address_a, None, address_a):
            await _set_focal(pool, session_id, focal)
            _sid, _eid, _addr = await _post_inbound(pool, prefix, "chat-A", content="tick")
            expected_stamps.append(focal)

        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, session_id)
        # The first event (auto-created) is before our loop — drop it.
        observed = [e.focal_channel_at_arrival for e in events[-len(expected_stamps) :]]
        assert observed == expected_stamps

    async def test_concurrent_appends_stamp_focal_consistently(
        self, pool: Any, agent_id: str, env_id: str, vault_id: str
    ) -> None:
        """Concurrent appends serialize through the session row lock.

        If two inbounds arrive simultaneously while focal is A, both
        should stamp focal_at_arrival=A (not NULL, not each other's
        scrap).  This pins down that the focal read lives inside the
        same transaction that allocates the seq.
        """
        from aios.db import queries

        _connection_id, prefix = await _setup_inbound(pool, agent_id, env_id, vault_id)
        session_id, _e1, address = await _post_inbound(pool, prefix, "chat-C")
        await _set_focal(pool, session_id, address)

        async def _fire() -> None:
            await _post_inbound(pool, prefix, "chat-C", content=f"msg-{_uniq()}")

        await asyncio.gather(*(_fire() for _ in range(5)))

        async with pool.acquire() as conn:
            events = await queries.read_message_events(conn, session_id)
        # All 5 concurrent user events plus the setup one — each has focal=address.
        user_events = [
            e
            for e in events
            if e.data.get("role") == "user" and e.data.get("metadata", {}).get("channel") == address
        ]
        # The first user event was appended before focal was set; the
        # remaining 5 were after.  All "after" events must stamp focal.
        focal_stamps = [e.focal_channel_at_arrival for e in user_events[1:]]
        assert len(focal_stamps) == 5
        assert all(s == address for s in focal_stamps), focal_stamps


# ─── slice 5: switch_channel tool ───────────────────────────────────────────


@pytest.fixture
async def runtime_pool(pool: Any) -> AsyncIterator[Any]:
    """Install ``runtime.pool`` so tool handlers can use it directly.

    Unit-style direct-handler tests exercise
    ``switch_channel_handler(session_id, arguments)`` without running
    the full step function; the handler calls ``runtime.require_pool()``
    which needs ``runtime.pool`` populated.
    """
    import aios.tools  # noqa: F401 — trigger registry population
    from aios.harness import runtime

    prev = runtime.pool
    runtime.pool = pool
    yield pool
    runtime.pool = prev


async def _get_session_focal(pool: Any, session_id: str) -> str | None:
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT focal_channel FROM sessions WHERE id = $1", session_id)


class TestSwitchChannelHandler:
    async def test_switch_to_target_updates_session_focal(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.tools.switch_channel import switch_channel_handler

        _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
        session_id, _e, address = await _post_inbound(runtime_pool, prefix, "chat-1")

        result = await switch_channel_handler(session_id, {"target": address})

        assert result.is_error is False
        assert await _get_session_focal(runtime_pool, session_id) == address
        assert result.metadata is not None
        marker = result.metadata["switch_channel"]
        assert marker == {"target": address, "success": True}

    async def test_switch_to_none_clears_focal(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.tools.switch_channel import switch_channel_handler

        _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
        session_id, _e, address = await _post_inbound(runtime_pool, prefix, "chat-1")
        await _set_focal(runtime_pool, session_id, address)

        result = await switch_channel_handler(session_id, {"target": None})

        assert result.is_error is False
        assert await _get_session_focal(runtime_pool, session_id) is None
        assert result.content == "Focal cleared."
        assert result.metadata is not None
        assert result.metadata["switch_channel"] == {"target": None, "success": True}

    async def test_switch_to_unknown_target_errors_and_keeps_focal(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.tools.switch_channel import switch_channel_handler

        _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
        session_id, _e, address = await _post_inbound(runtime_pool, prefix, "chat-1")
        await _set_focal(runtime_pool, session_id, address)  # focus on A

        result = await switch_channel_handler(session_id, {"target": "signal/other/fake"})

        assert result.is_error is True
        # Focal is unchanged after an invalid switch.
        assert await _get_session_focal(runtime_pool, session_id) == address
        assert result.metadata is not None
        marker = result.metadata["switch_channel"]
        assert marker == {"target": "signal/other/fake", "success": False}

    async def test_reorient_block_renders_recent_events_with_header(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.tools.switch_channel import switch_channel_handler

        _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
        session_id, _e, address = await _post_inbound(
            runtime_pool, prefix, "chat-1", content="first"
        )
        # Add more messages on this channel.
        for i in range(3):
            await _post_inbound(runtime_pool, prefix, "chat-1", content=f"msg-{i}")

        result = await switch_channel_handler(session_id, {"target": address})

        content = result.content
        assert isinstance(content, str)
        assert content.startswith(f"Switched to {address}. Recent messages:")
        # All 4 messages should appear (under the FLOOR_N=10 floor).
        for text in ("first", "msg-0", "msg-1", "msg-2"):
            assert text in content
        # The header convention signals focal rendering was used.
        assert f"[channel={address}" in content

    async def test_reorient_block_respects_floor_on_quiet_channel(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        """A switch into a channel with only 2 prior messages still
        shows those 2 (below the floor — no padding from nowhere)."""
        from aios.tools.switch_channel import switch_channel_handler

        _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
        session_id, _e, address = await _post_inbound(
            runtime_pool, prefix, "chat-1", content="only-one"
        )
        await _post_inbound(runtime_pool, prefix, "chat-1", content="only-two")

        result = await switch_channel_handler(session_id, {"target": address})
        content = result.content
        assert isinstance(content, str)
        assert "only-one" in content
        assert "only-two" in content

    async def test_reorient_block_includes_all_unread_when_over_floor(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        """Unread > FLOOR_N → all unread included, not clamped."""
        from aios.tools.switch_channel import RE_ORIENT_FLOOR_N, switch_channel_handler

        _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
        session_id, _e, address = await _post_inbound(runtime_pool, prefix, "chat-1")
        # Post MANY messages while focal is NULL so they all count as unread.
        n = RE_ORIENT_FLOOR_N + 5
        for i in range(n):
            await _post_inbound(runtime_pool, prefix, "chat-1", content=f"unread-{i:02d}")

        result = await switch_channel_handler(session_id, {"target": address})
        content = result.content
        assert isinstance(content, str)
        # Every unread message body should appear in the re-orient block.
        for i in range(n):
            assert f"unread-{i:02d}" in content

    async def test_reorient_block_empty_channel_fallback(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        """Switching into a bound channel that has never received a
        message shows a fallback line, not an empty block.
        """
        from aios.services import channels as ch_svc
        from aios.services import sessions as sess_svc
        from aios.tools.switch_channel import switch_channel_handler

        _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
        # Create a session + bind an unused channel address to it.
        session_id, _e, _addr_used = await _post_inbound(
            runtime_pool, prefix, "chat-used", content="seed"
        )
        unused_address = f"{prefix}/chat-never"
        await ch_svc.create_binding(runtime_pool, address=unused_address, session_id=session_id)
        # Sanity check session exists.
        await sess_svc.get_session(runtime_pool, session_id)

        result = await switch_channel_handler(session_id, {"target": unused_address})
        content = result.content
        assert isinstance(content, str)
        assert "no prior messages on this channel" in content
        assert await _get_session_focal(runtime_pool, session_id) == unused_address


class TestSwitchChannelAsEvent:
    """Switch_channel's tool_result event, as persisted by the dispatch
    path, must carry the ``metadata.switch_channel`` marker so unread
    derivation (slice 3) can use it as an anchor.
    """

    async def test_event_dispatch_persists_marker_metadata(
        self,
        harness: Any,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.db import queries
        from aios.harness.tool_dispatch import launch_tool_calls

        _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
        session_id, _e, address = await _post_inbound(runtime_pool, prefix, "chat-1")

        # Simulate an assistant message + a switch_channel tool call —
        # launch_tool_calls will invoke the handler and persist the
        # tool_result event via the dispatch path (exercising ToolResult).
        async with runtime_pool.acquire() as conn:
            await queries.append_event(
                conn,
                session_id=session_id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_switch_1",
                            "type": "function",
                            "function": {
                                "name": "switch_channel",
                                "arguments": json.dumps({"target": address}),
                            },
                        }
                    ],
                },
            )
        launch_tool_calls(
            runtime_pool,
            session_id,
            [
                {
                    "id": "call_switch_1",
                    "type": "function",
                    "function": {
                        "name": "switch_channel",
                        "arguments": json.dumps({"target": address}),
                    },
                }
            ],
        )
        # Wait for the fire-and-forget tool task to complete.  The task
        # registry tracks in-flight tasks by session; poll its count
        # with a small sleep budget.
        for _ in range(200):
            if harness._task_registry.in_flight_count(session_id) == 0:
                break
            await asyncio.sleep(0.01)
        assert harness._task_registry.in_flight_count(session_id) == 0

        async with runtime_pool.acquire() as conn:
            events = await queries.read_message_events(conn, session_id)

        tool_events = [
            e
            for e in events
            if e.data.get("role") == "tool" and e.data.get("name") == "switch_channel"
        ]
        assert len(tool_events) == 1
        ev = tool_events[0]
        # The marker lands under data.metadata — consumed by
        # derive_last_seen / derive_unread_counts in slice 3.
        assert ev.data.get("metadata", {}).get("switch_channel") == {
            "target": address,
            "success": True,
        }
        # Focal actually got flipped.
        assert await _get_session_focal(runtime_pool, session_id) == address


class TestMcpMetaInjection:
    """Slice 6: connection-provided MCP tools receive the focal channel
    path via the JSON-RPC ``_meta`` field, without stuffing it into
    arguments.  Agent-declared MCP servers don't get the meta stamp.
    """

    async def test_conn_tool_dispatch_injects_focal_suffix_into_meta(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from unittest.mock import AsyncMock, patch

        from aios.crypto.vault import CryptoBox
        from aios.harness import runtime
        from aios.harness.tool_dispatch import _execute_mcp_tool_async

        # The dispatch path calls resolve_auth_for_url which uses
        # runtime.crypto_box.
        prev_crypto = runtime.crypto_box
        runtime.crypto_box = CryptoBox(__import__("os").urandom(32))
        try:
            _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
            session_id, _e, address = await _post_inbound(runtime_pool, prefix, "chat-1")

            # The connection created by _setup_inbound has id prefix `conn_`.
            from aios.services import connections as conn_svc

            connections = await conn_svc.list_connections(runtime_pool)
            conn = next(c for c in connections if c.connector == "signal")

            mcp_server_map = {conn.id: conn.mcp_url}
            tool_call_dict = {
                "id": "call_send_1",
                "type": "function",
                "function": {
                    "name": f"mcp__{conn.id}__signal_send",
                    "arguments": json.dumps({"text": "hi there"}),
                },
            }

            with (
                patch(
                    "aios.mcp.client.resolve_auth_for_url",
                    new=AsyncMock(return_value={}),
                ),
                patch(
                    "aios.mcp.client.call_mcp_tool",
                    new=AsyncMock(return_value={"content": "ok"}),
                ) as mock_call,
            ):
                await _execute_mcp_tool_async(
                    runtime_pool,
                    session_id,
                    tool_call_dict,
                    mcp_server_map,
                    focal_channel=address,  # focal=A → suffix=chat-1
                )

            # Verify call_mcp_tool was invoked with the focal suffix meta.
            _args, kwargs = mock_call.call_args
            meta = kwargs.get("meta")
            assert isinstance(meta, dict)
            assert meta == {"aios.focal_channel_path": "chat-1"}
        finally:
            runtime.crypto_box = prev_crypto

    async def test_agent_mcp_tool_dispatch_no_meta_stamped(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        """Agent-declared MCP servers (not under the conn_* prefix) don't
        get focal meta — aios has no business telling an agent-declared
        MCP server about the session's focal channel.
        """
        from unittest.mock import AsyncMock, patch

        from aios.crypto.vault import CryptoBox
        from aios.harness import runtime
        from aios.harness.tool_dispatch import _execute_mcp_tool_async

        prev_crypto = runtime.crypto_box
        runtime.crypto_box = CryptoBox(__import__("os").urandom(32))
        try:
            _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)
            session_id, _e, address = await _post_inbound(runtime_pool, prefix, "chat-1")

            # Agent-declared server: name does NOT start with conn_.
            agent_server_name = "github"
            mcp_server_map = {agent_server_name: "https://mcp.github.com"}
            tool_call_dict = {
                "id": "call_gh_1",
                "type": "function",
                "function": {
                    "name": f"mcp__{agent_server_name}__create_issue",
                    "arguments": json.dumps({"title": "bug"}),
                },
            }

            with (
                patch(
                    "aios.mcp.client.resolve_auth_for_url",
                    new=AsyncMock(return_value={}),
                ),
                patch(
                    "aios.mcp.client.call_mcp_tool",
                    new=AsyncMock(return_value={"content": "done"}),
                ) as mock_call,
            ):
                await _execute_mcp_tool_async(
                    runtime_pool,
                    session_id,
                    tool_call_dict,
                    mcp_server_map,
                    focal_channel=address,  # non-null focal, but agent server
                )

            _args, kwargs = mock_call.call_args
            # No meta for agent-declared MCP.
            assert kwargs.get("meta") is None
        finally:
            runtime.crypto_box = prev_crypto


class TestOraSmokeTestRegression:
    """Regression for the smoke-test bug that motivated this redesign.

    The original failure: in a many-to-one session, a follow-up on a
    quiet DM was indistinguishable from questions about other channels'
    noise, because every channel's messages were interleaved.  Under
    the focal model, switching back to the DM surfaces its last-N
    messages (including the referent the follow-up points at) in the
    re-orient block — the agent has the context it needs.

    Design validation:
    * the re-orient block contains the "I'm Ora" claim (seeded early).
    * the re-orient block contains the follow-up "you sure about that?"
      (which arrived while agent was focused elsewhere).
    * focal_channel is set to the DM afterward, so the next inbound
      stamps correctly.
    """

    async def test_switch_back_to_quiet_channel_surfaces_referent(
        self,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.services import channels as ch_svc
        from aios.tools.switch_channel import switch_channel_handler

        _conn_id, prefix = await _setup_inbound(runtime_pool, agent_id, env_id, vault_id)

        # DM with Tom — session seeded here.
        session_id, _e_seed, dm_address = await _post_inbound(
            runtime_pool, prefix, "dm-tom", content="hey"
        )
        # Agent is focused on dm-tom, claims "I'm Ora".
        await _set_focal(runtime_pool, session_id, dm_address)

        # Inbound: the assistant self-identifies — in a real system this
        # is an assistant event, but for regression purposes we seed a
        # user event carrying the claim so it appears in the re-orient
        # block later.  (A real assistant message would already live in
        # the focal-native log and be visible to build_messages.)
        async with runtime_pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET focal_channel = $1 WHERE id = $2",
                dm_address,
                session_id,
            )
        from aios.services import sessions as sess_svc

        await sess_svc.append_user_message(
            runtime_pool,
            session_id,
            "I'm Ora — an AI running on this Signal channel.",
            metadata={"channel": dm_address, "sender_name": "Ora-claim"},
        )

        # Now agent switches to a second channel + burst of activity.
        qa_address = f"{prefix}/qa-group"
        await ch_svc.create_binding(runtime_pool, address=qa_address, session_id=session_id)
        await _set_focal(runtime_pool, session_id, qa_address)
        for i in range(30):
            await _post_inbound(runtime_pool, prefix, "qa-group", content=f"noise-{i:02d}")

        # Agent puts phone down (focal=None) and user sends the DM
        # follow-up while attention is elsewhere.
        await _set_focal(runtime_pool, session_id, None)
        await _post_inbound(runtime_pool, prefix, "dm-tom", content="you sure about that?")

        # Agent switches back to the DM.  Re-orient block must include
        # BOTH the "I'm Ora" seed and the "you sure" follow-up.
        result = await switch_channel_handler(session_id, {"target": dm_address})
        content = result.content
        assert isinstance(content, str)
        assert "I'm Ora" in content, f"missing referent in re-orient block:\n{content}"
        assert "you sure about that?" in content, (
            f"missing follow-up in re-orient block:\n{content}"
        )
        # Focal is set now; the DM-native view is active.
        assert await _get_session_focal(runtime_pool, session_id) == dm_address


def _msg_text(msg: dict[str, Any]) -> str:
    """Extract plain text from a chat-completions message.

    Messages can carry ``content`` as a plain string or a list of
    content blocks (Anthropic-style) — LiteLLM's cache-breakpoint
    injection turns the system-prompt string into a list-of-blocks
    shape.  Collapse both to plain text for substring assertions.
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict):
                t = b.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


class TestTailBlockInStep:
    """Slice 7: the ephemeral channels tail block appears as the last
    user-role message on the chat-completions list, and its unread
    counts update across steps without busting the cache-stable
    system prompt.
    """

    async def test_tail_block_appears_as_last_user_message(
        self,
        harness: Any,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.services import channels as ch_svc
        from aios.services import connections as conn_svc
        from tests.e2e.harness import assistant

        # Build a session with a routed inbound so bindings exist.
        account = f"tail-{_uniq()}"
        connection = await conn_svc.create_connection(
            runtime_pool,
            connector="signal",
            account=account,
            mcp_url="https://m",
            vault_id=vault_id,
            metadata={},
        )
        await ch_svc.create_routing_rule(
            runtime_pool,
            prefix=f"signal/{account}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        prefix = f"signal/{account}"

        # Route one inbound to auto-create a session.
        address = f"{prefix}/chat-1"
        with mock.patch("aios.harness.wake.defer_wake"):
            resolution = await ch_svc.resolve_channel(runtime_pool, address)
            from aios.services import sessions as sess_svc

            await sess_svc.append_user_message(
                runtime_pool,
                resolution.session_id,
                "hi",
                metadata={"channel": address},
            )
        session_id = resolution.session_id
        # Ignore the connection — the channel listing pulls from bindings.
        assert connection.id.startswith("conn_")

        harness.script_model([assistant("ok")])
        await harness.run_step(session_id)

        calls = harness.model_calls
        assert calls, "litellm was not called"
        messages = calls[-1]["messages"]
        # Tail block is the last user message in the payload.
        assert messages[-1]["role"] == "user"
        content = _msg_text(messages[-1])
        assert "━━━ Channels ━━━" in content
        # NULL focal at this point → no ▸ marker, the one binding appears as ○.
        assert f"○ {address}" in content
        assert "▸" not in content

    async def test_tail_block_reflects_focal_and_unread_changes(
        self,
        harness: Any,
        runtime_pool: Any,
        agent_id: str,
        env_id: str,
        vault_id: str,
    ) -> None:
        from aios.services import channels as ch_svc
        from aios.services import connections as conn_svc
        from tests.e2e.harness import assistant

        account = f"tail2-{_uniq()}"
        await conn_svc.create_connection(
            runtime_pool,
            connector="signal",
            account=account,
            mcp_url="https://m",
            vault_id=vault_id,
            metadata={},
        )
        await ch_svc.create_routing_rule(
            runtime_pool,
            prefix=f"signal/{account}",
            target=f"agent:{agent_id}",
            session_params=SessionParams(environment_id=env_id),
        )
        prefix = f"signal/{account}"

        # Route two channels to the same session.
        address_a = f"{prefix}/chat-A"
        address_b = f"{prefix}/chat-B"
        with mock.patch("aios.harness.wake.defer_wake"):
            resolution_a = await ch_svc.resolve_channel(runtime_pool, address_a)
            session_id = resolution_a.session_id
            await ch_svc.create_binding(runtime_pool, address=address_b, session_id=session_id)
            # Focus on A and let one message land in A.
            from aios.services import sessions as sess_svc

            await _set_focal(runtime_pool, session_id, address_a)
            await sess_svc.append_user_message(
                runtime_pool,
                session_id,
                "hi from A",
                metadata={"channel": address_a},
            )
            # Then two messages on B while focal is A → unread in B.
            await sess_svc.append_user_message(
                runtime_pool,
                session_id,
                "msg-b1",
                metadata={"channel": address_b},
            )
            await sess_svc.append_user_message(
                runtime_pool,
                session_id,
                "msg-b2",
                metadata={"channel": address_b},
            )

        # Step 1: focal=A.  Tail block should mark A as focal, show 2 unread on B.
        harness.script_model([assistant("processing")])
        await harness.run_step(session_id)
        messages = harness.model_calls[-1]["messages"]
        assert messages[-1]["role"] == "user"
        content_step1 = _msg_text(messages[-1])
        assert f"▸ {address_a} (focal)" in content_step1
        assert f"○ {address_b} — 2 unread" in content_step1
        assert "msg-b2" in content_step1  # preview of latest unread

        # Step 2: change focal to B and run again.  Tail block reflects.
        await _set_focal(runtime_pool, session_id, address_b)
        await sess_svc.append_user_message(
            runtime_pool,
            session_id,
            "more activity",
            metadata={"channel": address_b},
        )
        harness.script_model([assistant("noted")])
        await harness.run_step(session_id)
        content_step2 = _msg_text(harness.model_calls[-1]["messages"][-1])
        assert f"▸ {address_b} (focal)" in content_step2
        # A is now non-focal; no unread for A yet since we never switched
        # away after its last focal-stamp, but the listing should show it.
        assert f"○ {address_a}" in content_step2
