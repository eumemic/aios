"""Unit tests for ``services.sessions._classify_awaiting`` (#816).

The read-model classifier turns an unresolved-tool_call entry dict (as
produced by ``list_unresolved_tool_calls_batch``) into an
``AwaitingToolCall`` for ``Session.awaiting``. #816 added a required
``pending_since`` field — the declaring assistant event's ``created_at``
— so clients can distinguish a healthy in-flight custom call from a stuck
one. These tests assert ``pending_since`` propagates onto the classified
entry for each of the custom / builtin / mcp branches.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from aios.models.agents import AgentBinding, StepSurface, ToolSpec
from aios.models.sessions import AwaitingToolCall
from aios.services.sessions import _classify_awaiting

PENDING_SINCE = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)


def _make_agent(tools: list[ToolSpec]) -> StepSurface:
    return StepSurface(
        model="gpt-test",
        system="be helpful",
        tools=tools,
        skills=[],
        mcp_servers=[],
        http_servers=[],
        litellm_extra={},
        window_min=1,
        window_max=10,
        binding=AgentBinding(agent_id="agt_test", version=1),
    )


def _entry(name: str, *, has_allow_lifecycle: bool = False, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "tool_call_id": "tc_1",
        "name": name,
        "arguments": "{}",
        "has_allow_lifecycle": has_allow_lifecycle,
        "pending_since": PENDING_SINCE,
    }
    entry.update(overrides)
    return entry


class TestClassifyAwaitingPendingSince:
    def test_custom_tool_carries_pending_since(self) -> None:
        agent = _make_agent([])
        result = _classify_awaiting(_entry("some_custom_connector_tool"), agent)
        assert result is not None
        assert result.kind == "custom"
        assert result.pending_since == PENDING_SINCE

    def test_builtin_always_ask_carries_pending_since(self) -> None:
        agent = _make_agent([ToolSpec(type="bash", permission="always_ask")])
        result = _classify_awaiting(_entry("bash"), agent)
        assert result is not None
        assert result.kind == "builtin"
        assert result.pending_since == PENDING_SINCE

    def test_mcp_always_ask_carries_pending_since(self) -> None:
        # No matching mcp_toolset entry → effective permission defaults to
        # always_ask, so the mcp call surfaces as awaiting.
        agent = _make_agent([])
        result = _classify_awaiting(_entry("mcp__server__tool"), agent)
        assert result is not None
        assert result.kind == "mcp"
        assert result.pending_since == PENDING_SINCE


class TestAwaitingToolCallModel:
    def test_pending_since_is_required(self) -> None:
        with pytest.raises(ValidationError):
            AwaitingToolCall(tool_call_id="tc_1", name="x", kind="custom")  # type: ignore[call-arg]

    def test_constructs_with_pending_since(self) -> None:
        call = AwaitingToolCall(
            tool_call_id="tc_1", name="x", kind="custom", pending_since=PENDING_SINCE
        )
        assert call.pending_since == PENDING_SINCE
