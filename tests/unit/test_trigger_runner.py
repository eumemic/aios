from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aios.harness import trigger_runner
from aios.models.triggers import SandboxCommandAction


@pytest.mark.asyncio
async def test_sandbox_observation_is_finished_on_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = Mock()
    registry = Mock()
    registry.get_or_provision.side_effect = asyncio.CancelledError

    monkeypatch.setattr(trigger_runner.runtime, "require_pool", Mock(return_value=Mock()))
    monkeypatch.setattr(
        trigger_runner.runtime, "require_sandbox_registry", Mock(return_value=registry)
    )
    monkeypatch.setattr(trigger_runner.runtime, "require_tool_broker", Mock(return_value=broker))

    trigger = SimpleNamespace(owner_session_id="session-1")
    action = SandboxCommandAction(command="true")

    with pytest.raises(asyncio.CancelledError):
        await trigger_runner._run_sandbox_command(trigger, action)  # type: ignore[arg-type]

    token = broker.begin_trigger_observation.call_args.args[0]
    broker.finish_trigger_observation.assert_called_once_with(token)
