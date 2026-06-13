"""Unit tests for ``services.sessions._classify_awaiting`` — the in-memory
classifier that turns an unresolved-tool_call dict into an ``AwaitingToolCall``.

#816: every classified entry must carry ``pending_since`` (the declaring
assistant event's ``created_at``), so clients can age custom calls to tell a
healthy in-flight call from a stuck one without loading the transcript.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from aios.models.agents import Agent, AgentVersion, ToolSpec
from aios.services.sessions import _classify_awaiting

PENDING_SINCE = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)


def _agent(*, tools: list[ToolSpec] | None = None) -> Agent | AgentVersion:
    return cast(Agent | AgentVersion, SimpleNamespace(tools=tools or [], http_servers=[]))


def _tc(name: str, *, has_allow_lifecycle: bool = False) -> dict[str, Any]:
    return {
        "tool_call_id": "tc_1",
        "name": name,
        "arguments": "{}",
        "has_allow_lifecycle": has_allow_lifecycle,
        "pending_since": PENDING_SINCE,
    }


class TestClassifyAwaitingPendingSince:
    def test_custom_tool_carries_pending_since(self) -> None:
        agent = _agent()
        result = _classify_awaiting(_tc("some_client_tool"), agent)
        assert result is not None
        assert result.kind == "custom"
        assert result.pending_since == PENDING_SINCE

    def test_always_ask_builtin_carries_pending_since(self) -> None:
        agent = _agent(tools=[ToolSpec(type="bash", permission="always_ask")])
        result = _classify_awaiting(_tc("bash"), agent)
        assert result is not None
        assert result.kind == "builtin"
        assert result.pending_since == PENDING_SINCE
