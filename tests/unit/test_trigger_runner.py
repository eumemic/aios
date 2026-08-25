from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aios.db import queries
from aios.harness import runtime, trigger_runner
from aios.models.triggers import SandboxCommandAction, WakeOwnerAction
from aios.sandbox.tool_broker import ToolBroker


@pytest.mark.asyncio
async def test_sandbox_observation_is_finished_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = ToolBroker()
    registry = Mock()
    registry.get_or_provision.side_effect = asyncio.CancelledError

    monkeypatch.setattr(runtime, "require_pool", Mock(return_value=Mock()))
    monkeypatch.setattr(runtime, "require_sandbox_registry", Mock(return_value=registry))
    monkeypatch.setattr(runtime, "require_tool_broker", Mock(return_value=broker))

    trigger = SimpleNamespace(owner_session_id="session-1")
    action = SandboxCommandAction(command="true")

    with pytest.raises(asyncio.CancelledError):
        await trigger_runner._run_sandbox_command(trigger, action)  # type: ignore[arg-type]

    assert broker._trigger_observations == {}


@pytest.mark.asyncio
async def test_observation_reader_failure_does_not_fail_completed_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Connection:
        @asynccontextmanager
        async def transaction(self):
            yield self

    class Pool:
        @asynccontextmanager
        async def acquire(self):
            yield Connection()

    trigger = SimpleNamespace(
        id="trigger-1",
        source="cron",
        action=WakeOwnerAction(content="wake"),
        session_archived_at=None,
        enabled=True,
        owner_session_id="owner",
        account_id="account",
        name="watchdog",
        source_spec={},
    )
    deliver = AsyncMock(return_value=("ok", None, None))

    monkeypatch.setattr(runtime, "require_pool", Mock(return_value=Pool()))
    monkeypatch.setattr(queries, "unscoped_get_trigger_row", AsyncMock(return_value=trigger))
    monkeypatch.setattr(trigger_runner, "_run_wake_owner", deliver)
    monkeypatch.setattr(queries, "record_trigger_fire", AsyncMock(return_value=0))
    monkeypatch.setattr(trigger_runner, "_record_timer_audit", AsyncMock(return_value="audit"))
    monkeypatch.setattr(trigger_runner, "_append_fire_event", AsyncMock())
    monkeypatch.setattr(
        queries,
        "list_recent_trigger_wake_outcomes",
        AsyncMock(side_effect=RuntimeError("telemetry reader unavailable")),
    )

    await trigger_runner.run_trigger_step("trigger-1")

    deliver.assert_awaited_once()
