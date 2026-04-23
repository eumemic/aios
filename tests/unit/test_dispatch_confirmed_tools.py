"""Unit tests for ``_dispatch_confirmed_tools`` tool-confirmation lookup."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.loop import _dispatch_confirmed_tools


def _assistant_with_tool_calls(tool_call_ids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        kind="message",
        data={
            "role": "assistant",
            "tool_calls": [{"id": tcid, "type": "function"} for tcid in tool_call_ids],
        },
    )


def _confirmed(tool_call_id: str, result: str = "allow") -> SimpleNamespace:
    return SimpleNamespace(
        data={"event": "tool_confirmed", "result": result, "tool_call_id": tool_call_id}
    )


class TestDispatchConfirmedTools:
    async def test_returns_empty_when_no_assistant_tool_calls(self) -> None:
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=[]),
        ):
            assert await _dispatch_confirmed_tools(pool, "sess_x", []) == []

    async def test_returns_confirmed_but_not_completed(self) -> None:
        """Baseline: a confirmed tool call with no tool result is pending."""
        msg_events = [_assistant_with_tool_calls(["tc1"])]
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=[_confirmed("tc1")]),
        ):
            pending = await _dispatch_confirmed_tools(pool, "sess_x", msg_events)
        assert [tc["id"] for tc in pending] == ["tc1"]

    async def test_reads_lifecycle_tail_newest_first(self) -> None:
        """Regression for #155: default ASC + LIMIT 200 scan drops recent
        ``tool_confirmed`` events on long sessions (bulk are ancient
        ``turn_ended``). The user clicks allow but the confirmation is
        invisible to the dispatch sweep.
        """
        msg_events = [_assistant_with_tool_calls(["tc1"])]
        mock_read = AsyncMock(return_value=[_confirmed("tc1")])
        pool = MagicMock()
        with patch("aios.harness.loop.sessions_service.read_events", mock_read):
            pending = await _dispatch_confirmed_tools(pool, "sess_x", msg_events)
        assert [tc["id"] for tc in pending] == ["tc1"]
        assert mock_read.call_args.kwargs["newest_first"] is True
