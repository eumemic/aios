"""Unit tests for the ``wake_self`` tool handler.

The handler appends a user-role message to the *same* session that
invoked it and defers a wake. It exists so cron-fired bash commands
(and any other sandbox-side caller) can hand a user-role prompt to
their own next model step without curling the broker.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from jsonschema import ValidationError, validate

from aios.tools.wake_self import (
    WAKE_SELF_PARAMETERS_SCHEMA,
    WakeSelfArgumentError,
    wake_self_handler,
)


@pytest.fixture(autouse=True)
def mock_runtime_pool(monkeypatch: Any) -> MagicMock:
    """Stub the worker-only runtime pool so the handler can resolve
    ``runtime.require_pool()`` without a live worker_main context.
    """
    pool = MagicMock()
    monkeypatch.setattr("aios.tools.wake_self.runtime.require_pool", lambda: pool)
    return pool


@pytest.fixture
def mock_services(monkeypatch: Any) -> dict[str, AsyncMock]:
    """Replace the session/wake service calls with AsyncMocks so the
    handler runs without Postgres.

    Returns the mocks keyed by name for per-test assertions.
    """
    load_account = AsyncMock(return_value="acct_TEST")
    event = MagicMock(id="evt_TEST")
    append = AsyncMock(return_value=event)
    defer = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "aios.tools.wake_self.sessions_service.load_session_account_id",
        load_account,
    )
    monkeypatch.setattr(
        "aios.tools.wake_self.sessions_service.append_user_message",
        append,
    )
    monkeypatch.setattr("aios.tools.wake_self.defer_wake", defer)
    return {
        "load_account": load_account,
        "append": append,
        "defer": defer,
    }


class TestWakeSelfHandler:
    async def test_handler_appends_user_message_and_defers_wake(
        self,
        mock_runtime_pool: MagicMock,
        mock_services: dict[str, AsyncMock],
    ) -> None:
        result = await wake_self_handler("sess_TEST", {"content": "ping"})

        mock_services["load_account"].assert_awaited_once_with(mock_runtime_pool, "sess_TEST")
        mock_services["append"].assert_awaited_once_with(
            mock_runtime_pool, "sess_TEST", "ping", account_id="acct_TEST"
        )
        mock_services["defer"].assert_awaited_once_with(
            mock_runtime_pool, "sess_TEST", cause="message", account_id="acct_TEST"
        )
        assert result == {
            "woken": True,
            "session_id": "sess_TEST",
            "event_id": "evt_TEST",
        }

    async def test_empty_content_rejected(self, mock_services: dict[str, AsyncMock]) -> None:
        with pytest.raises(WakeSelfArgumentError):
            await wake_self_handler("sess_TEST", {"content": ""})
        mock_services["append"].assert_not_awaited()
        mock_services["defer"].assert_not_awaited()

    @pytest.mark.parametrize("bad_content", [None, 42, ["a"], {"k": "v"}])
    async def test_non_string_content_rejected(
        self,
        mock_services: dict[str, AsyncMock],
        bad_content: Any,
    ) -> None:
        with pytest.raises(WakeSelfArgumentError):
            await wake_self_handler("sess_TEST", {"content": bad_content})
        mock_services["append"].assert_not_awaited()
        mock_services["defer"].assert_not_awaited()

    async def test_missing_content_rejected(self, mock_services: dict[str, AsyncMock]) -> None:
        with pytest.raises(WakeSelfArgumentError):
            await wake_self_handler("sess_TEST", {})
        mock_services["append"].assert_not_awaited()
        mock_services["defer"].assert_not_awaited()


class TestWakeSelfSchema:
    def test_additional_properties_rejected_by_schema(self) -> None:
        with pytest.raises(ValidationError):
            validate(
                {"content": "x", "metadata": {}},
                WAKE_SELF_PARAMETERS_SCHEMA,
            )


class TestWakeSelfRegistration:
    def test_registered_with_correct_metadata(self) -> None:
        # Importing the tools package triggers all _register() side-effects.
        import aios.tools  # noqa: F401
        from aios.tools.registry import registry

        tool = registry.get("wake_self")
        assert tool.transport == "both"
        assert tool.parameters_schema == WAKE_SELF_PARAMETERS_SCHEMA
        assert tool.description


class TestWakeSelfToolSpec:
    """Regression: ``wake_self`` must be declarable on an agent.

    Without ``"wake_self"`` in ``BuiltinToolType``, Pydantic rejects
    ``ToolSpec(type="wake_self")``, agents cannot list the tool, and
    the broker's ``_find_builtin_spec`` 404s every CLI invocation —
    defeating the entire purpose of the tool.
    """

    def test_toolspec_accepts_wake_self(self) -> None:
        from aios.models.agents import ToolSpec

        spec = ToolSpec(type="wake_self")
        assert spec.type == "wake_self"

    def test_wake_self_in_builtin_names(self) -> None:
        from aios.models.agents import _BUILTIN_NAMES

        assert "wake_self" in _BUILTIN_NAMES
