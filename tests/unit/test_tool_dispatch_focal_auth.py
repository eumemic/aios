"""Unit test for focal-route authorization in MCP outbound dispatch.

The bug: when a model calls a connector tool that takes ``account`` from
the focal channel meta (rather than as an explicit arg), the dispatch
path skipped ``validate_account_for_session`` entirely.  Result: a
session attached to one connector could send via *any* attached
connector simply by switching focal to a channel address it didn't own.

Demonstrated live during PR #213 migration testing: the factchecker
session (telegram-only) ghost-wrote a "Testing Signal send" message
into the Metals and AI Signal group despite having no signal
connection attached.

This test pins down the expected behaviour: focal-route dispatch MUST
call ``validate_account_for_session`` and short-circuit with a
tool_error if the session isn't authorized for the focal account.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness import runtime, tool_dispatch
from aios.services import connections as connections_service
from aios.services import sessions as sessions_service


def _make_call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "tc1",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}),
        },
    }


@pytest.fixture
def mock_connector_registry(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stand in for runtime.connector_subprocess_registry with a single signal instance."""
    registry = MagicMock()
    state = MagicMock()
    state.instance = "signal"
    state.connector = "signal"
    registry.states_for_connector.return_value = [state]
    registry.dispatch_call_for_account = AsyncMock(return_value={"sent_at_ms": 1})
    registry.dispatch_call = AsyncMock(return_value={"sent_at_ms": 1})
    monkeypatch.setattr(runtime, "connector_subprocess_registry", registry)
    return registry


@pytest.fixture
def silenced_appenders(monkeypatch: pytest.MonkeyPatch) -> dict[str, AsyncMock]:
    """Replace sessions_service.append_event with an AsyncMock that records calls.

    Return value is a stand-in Event with ``.id`` and ``.seq`` so the
    tool_dispatch code reading ``span_start.id`` etc. doesn't blow up.
    """
    fake_event = MagicMock()
    fake_event.id = "evt_test"
    fake_event.seq = 1

    append = AsyncMock(return_value=fake_event)
    monkeypatch.setattr(sessions_service, "append_event", append)
    monkeypatch.setattr(tool_dispatch.sessions_service, "append_event", append)

    # _trigger_sweep imports sweep lazily; stub at the tool_dispatch level.
    sweep = AsyncMock(return_value=None)
    monkeypatch.setattr(tool_dispatch, "_trigger_sweep", sweep)

    return {"append_event": append, "sweep": sweep}


class TestFocalRouteAuthorization:
    """``_execute_mcp_tool_async`` must validate focal-derived accounts."""

    async def test_focal_routed_unauthorized_account_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_connector_registry: MagicMock,
        silenced_appenders: dict[str, AsyncMock],
    ) -> None:
        """Session with no signal attachment cannot send via focal-routed signal_send.

        Reproduces the live PR #213 finding: factchecker (telegram-only)
        ghost-writing into a Signal group via focal-route bypass.
        """
        validate = AsyncMock(return_value=False)  # session NOT authorized
        monkeypatch.setattr(connections_service, "validate_account_for_session", validate)

        pool = MagicMock()
        await tool_dispatch._execute_mcp_tool_async(
            pool=pool,
            session_id="sess_factchecker",
            call=_make_call("mcp__signal__signal_send", {"text": "ghost write"}),
            mcp_server_map={},
            focal_channel="signal/some-account-uuid/some-chat-id",
        )

        # Authorization MUST have been checked for the focal-derived account.
        validate.assert_awaited_once()
        assert validate.await_args is not None
        kwargs = validate.await_args.kwargs
        assert kwargs.get("connector") == "signal"
        assert kwargs.get("account") == "some-account-uuid"

        # Dispatch MUST NOT have been called.
        mock_connector_registry.dispatch_call_for_account.assert_not_awaited()
        mock_connector_registry.dispatch_call.assert_not_awaited()

        # A tool_error event MUST have been appended.
        append_calls = silenced_appenders["append_event"].await_args_list
        message_calls = [c for c in append_calls if len(c.args) >= 3 and c.args[2] == "message"]
        assert message_calls, "expected at least one 'message' append (the error)"
        last_message = message_calls[-1].args[3]
        assert last_message.get("role") == "tool"
        assert last_message.get("is_error") is True
        content = json.loads(last_message.get("content", "{}"))
        assert "not attached" in content.get("error", "")

    async def test_focal_routed_authorized_account_dispatches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_connector_registry: MagicMock,
        silenced_appenders: dict[str, AsyncMock],
    ) -> None:
        """Session WITH attached connection on focal account dispatches normally."""
        validate = AsyncMock(return_value=True)  # session IS authorized
        monkeypatch.setattr(connections_service, "validate_account_for_session", validate)

        pool = MagicMock()
        await tool_dispatch._execute_mcp_tool_async(
            pool=pool,
            session_id="sess_factchecker",
            call=_make_call("mcp__signal__signal_send", {"text": "legit"}),
            mcp_server_map={},
            focal_channel="signal/owned-account/some-chat-id",
        )

        validate.assert_awaited_once()
        mock_connector_registry.dispatch_call_for_account.assert_awaited_once()
