"""E2E coverage for the focal-channel attention model.

Replaces the migration-era ``test_focal_channel.py`` (deleted with the
old ``channel_bindings`` table).  Bound channels are now derived from
``events.channel`` stamps; an inbound with ``metadata.channel`` set
populates the column at append time.

Coverage:

* ``switch_channel`` happy path mutates ``sessions.focal_channel`` and
  emits the success marker on the tool_result.
* Unknown-target rejection — switching to a channel the session has
  never seen returns ``is_error`` and leaves focal unchanged.
* Per_chat-spawned sessions reject ``switch_channel`` calls regardless
  of target (focal-locked invariant).
* The cache-stable focal paradigm prose appears in the system prompt
  whenever the session has any bound channel; the per-step tail block
  appears as a user-role message and renders unread counts off the
  event log.
* Bare assistant text is auto-prefixed with the monologue marker once
  the session has bound channels.
"""

from __future__ import annotations

from typing import Any

from aios.db import queries
from aios.harness.channels import (
    FOCAL_CHANNEL_META_KEY,
    MONOLOGUE_PREFIX,
    SWITCH_CHANNEL_METADATA_KEY,
    focal_channel_path,
)
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, msg_text, tool_call

_TAIL_HEADER = "━━━ Channels ━━━"
_PARADIGM_HEADER = "## Channels & focal attention"


def _switch_call(target: str | None, *, call_id: str = "call_switch") -> dict[str, Any]:
    return tool_call("switch_channel", {"channel_id": target}, call_id=call_id)


@needs_docker
class TestFocalChannelE2E:
    async def test_switch_channel_mutates_focal_and_marks_result(self, harness: Harness) -> None:
        """Successful switch updates ``sessions.focal_channel`` and the
        tool_result carries the ``switch_channel`` success marker the
        unread-derivation helpers anchor on.
        """
        harness.script_model(
            [
                assistant(tool_calls=[_switch_call("signal/+1/chat-a", call_id="c1")]),
                assistant("ok"),
            ]
        )
        session = await harness.start("hi")
        # Two channel-stamped inbounds so the session is "bound" to both.
        await sessions_service.append_user_message(
            harness._pool,
            session.id,
            "from chat-a",
            metadata={"channel": "signal/+1/chat-a"},
        )
        await sessions_service.append_user_message(
            harness._pool,
            session.id,
            "from chat-b",
            metadata={"channel": "signal/+1/chat-b"},
        )
        await harness.run_until_idle(session.id)

        # Focal updated to the requested target.
        async with harness._pool.acquire() as conn:
            focal = await queries.get_session_focal_channel(conn, session.id)
        assert focal == "signal/+1/chat-a"

        # Tool result carries the success marker.
        events = await harness.events(session.id)
        tool_results = [e for e in events if e.data.get("role") == "tool"]
        assert tool_results, "expected at least one tool_result event"
        marker = tool_results[-1].data.get("metadata", {}).get(SWITCH_CHANNEL_METADATA_KEY)
        assert marker == {"target": "signal/+1/chat-a", "success": True}

    async def test_switch_channel_rejects_unknown_target(self, harness: Harness) -> None:
        """A target that the session has never received an event on is
        not a bound channel; the handler returns is_error and leaves
        focal unchanged.
        """
        harness.script_model(
            [
                assistant(tool_calls=[_switch_call("signal/+1/never-seen", call_id="c1")]),
                assistant("ok"),
            ]
        )
        session = await harness.start("hi")
        await sessions_service.append_user_message(
            harness._pool,
            session.id,
            "from chat-a",
            metadata={"channel": "signal/+1/chat-a"},
        )
        await harness.run_until_idle(session.id)

        async with harness._pool.acquire() as conn:
            focal = await queries.get_session_focal_channel(conn, session.id)
        assert focal is None, "rejected switch must not mutate focal"

        events = await harness.events(session.id)
        result = next(e for e in events if e.data.get("role") == "tool")
        assert result.data.get("is_error") is True
        marker = result.data.get("metadata", {}).get(SWITCH_CHANNEL_METADATA_KEY)
        assert marker == {"target": "signal/+1/never-seen", "success": False}

    async def test_per_chat_session_rejects_switch_channel(self, harness: Harness) -> None:
        """A session whose ``spawned_from_connection_id`` is set is bound
        to one chat by construction; ``switch_channel`` rejects every
        attempt regardless of target.
        """
        # Build a connection so we have a real id to point spawned_from at.
        from aios.services import connections as connections_service

        conn_row = await connections_service.create_connection(
            harness._pool,
            connector="signal",
            account="+1",
            metadata={},
        )

        harness.script_model(
            [
                assistant(tool_calls=[_switch_call("signal/+1/chat-a", call_id="c1")]),
                assistant("ok"),
            ]
        )
        # Spawn a session in per_chat mode via the same code path the
        # inbound handler will use in PR3.
        from aios.ids import make_id
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service

        if harness._env_id is None:
            from aios.services import environments as environments_service

            env = await environments_service.create_environment(
                harness._pool, name=f"focal-env-{make_id('env')[-8:]}"
            )
            harness._env_id = env.id
        agent = await agents_service.create_agent(
            harness._pool,
            name=f"focal-per-chat-{make_id('agent')[-8:]}",
            model="fake/test",
            system="test",
            tools=[ToolSpec(type="bash")],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=harness._env_id,
            title="per-chat",
            metadata={},
            spawned_from_connection_id=conn_row.id,
            focal_channel="signal/+1/chat-a",
        )
        await sessions_service.append_user_message(
            harness._pool,
            session.id,
            "hi from chat-a",
            metadata={"channel": "signal/+1/chat-a"},
        )
        await harness.run_until_idle(session.id)

        # Focal must be the spawn-time value, untouched.
        async with harness._pool.acquire() as conn:
            focal = await queries.get_session_focal_channel(conn, session.id)
        assert focal == "signal/+1/chat-a"

        events = await harness.events(session.id)
        result = next(e for e in events if e.data.get("role") == "tool")
        assert result.data.get("is_error") is True
        assert "per_chat" in result.data["content"]

    async def test_paradigm_block_renders_when_session_has_channels(self, harness: Harness) -> None:
        """The cache-stable focal-channel paradigm block appears in the
        system prompt once the session has any bound channel; the
        ephemeral tail block appears as a user-role message.
        """
        harness.script_model([assistant("ok")])
        session = await harness.start("hi")
        await sessions_service.append_user_message(
            harness._pool,
            session.id,
            "first inbound",
            metadata={"channel": "signal/+1/chat-a"},
        )
        await harness.run_until_idle(session.id)

        msgs = harness.model_calls[0]["messages"]
        # Paradigm prose lives in the system prompt for prefix-cache stability.
        assert msgs[0]["role"] == "system"
        assert _PARADIGM_HEADER in msgs[0]["content"]
        # Tail block lives as the trailing user message.
        tail = next(
            (m for m in msgs if m.get("role") == "user" and _TAIL_HEADER in msg_text(m)),
            None,
        )
        assert tail is not None, f"tail block not found: {msgs!r}"
        assert "channel_id=signal/+1/chat-a" in msg_text(tail)

    async def test_bare_assistant_text_gets_monologue_prefix(self, harness: Harness) -> None:
        """Bare assistant text on a channel-bearing session is monologue
        (the connector tools, not the text, deliver to the peer).  The
        loop applies ``MONOLOGUE_PREFIX`` so the log is uniform on replay.
        """
        harness.script_model([assistant("thinking out loud")])
        session = await harness.start("hi")
        await sessions_service.append_user_message(
            harness._pool,
            session.id,
            "channel inbound",
            metadata={"channel": "signal/+1/chat-a"},
        )
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        asst = next(e for e in events if e.data.get("role") == "assistant")
        content = asst.data.get("content")
        assert isinstance(content, str)
        assert content.startswith(MONOLOGUE_PREFIX), content


class TestFocalChannelPathHelper:
    """Pure unit tests for the helper that strips connector/account from
    a focal address before injection into MCP ``_meta``.  Lives here so
    the e2e file is the single home for focal-channel coverage; the
    helper is otherwise covered indirectly by the dispatch path.
    """

    def test_strips_connector_and_account(self) -> None:
        assert focal_channel_path("signal/+1/chat-a") == "chat-a"

    def test_preserves_trailing_segments(self) -> None:
        assert focal_channel_path("telegram/bot/group/thread-1") == "group/thread-1"

    def test_returns_none_when_focal_unset(self) -> None:
        assert focal_channel_path(None) is None

    def test_returns_none_for_two_segment_addresses(self) -> None:
        # Less than 3 segments means no chat suffix to inject.
        assert focal_channel_path("signal/+1") is None

    def test_meta_key_constant_is_stable(self) -> None:
        # External connectors snapshot this string; flag any rename.
        assert FOCAL_CHANNEL_META_KEY == "aios.focal_channel_path"
