"""Unit tests for the bash tool handler (with mocked container).

Exercises argument validation and result shaping. The sandbox registry
is stubbed so tests don't touch Docker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.harness import runtime
from aios.sandbox.backends.base import CommandResult, SandboxHandle
from aios.tools.bash import BashArgumentError, bash_handler


class _StubRegistry:
    """Minimal stand-in for SandboxRegistry used by handler tests."""

    def __init__(self, handle: SandboxHandle, result: CommandResult) -> None:
        self._handle = handle
        self.get_or_provision_calls: list[str] = []
        self.exec = AsyncMock(return_value=result)

    async def get_or_provision(self, session_id: str, **_kwargs: Any) -> SandboxHandle:
        self.get_or_provision_calls.append(session_id)
        return self._handle


@pytest.fixture
def stub_handle(**kwargs: Any) -> SandboxHandle:
    """A SandboxHandle with run_command mocked out."""
    handle = SandboxHandle(
        session_id="sess_01TEST",
        sandbox_id="container_abc123",
        workspace_path=Path("/tmp/aios-test-workspace"),
    )
    return handle


@pytest.fixture
def canned_result() -> CommandResult:
    return CommandResult(
        exit_code=0,
        stdout="hello world\n",
        stderr="",
        timed_out=False,
        truncated=False,
    )


@pytest.fixture
def stub_registry(
    stub_handle: SandboxHandle, canned_result: CommandResult, **kwargs: Any
) -> _StubRegistry:
    """Install a stub sandbox registry on the runtime module, restore after."""
    from unittest.mock import MagicMock

    prev_registry = runtime.sandbox_registry
    prev_pool = runtime.pool
    stub = _StubRegistry(stub_handle, canned_result)
    runtime.sandbox_registry = stub  # type: ignore[assignment]
    runtime.pool = MagicMock()
    try:
        yield stub
    finally:
        runtime.sandbox_registry = prev_registry
        runtime.pool = prev_pool


class TestArguments:
    async def test_rejects_missing_command(self, stub_registry: _StubRegistry) -> None:
        with pytest.raises(BashArgumentError):
            await bash_handler("sess_01TEST", {})

    async def test_rejects_empty_command(self, stub_registry: _StubRegistry) -> None:
        with pytest.raises(BashArgumentError):
            await bash_handler("sess_01TEST", {"command": "   "})

    async def test_rejects_non_string_command(self, stub_registry: _StubRegistry) -> None:
        with pytest.raises(BashArgumentError):
            await bash_handler("sess_01TEST", {"command": 123})

    async def test_rejects_zero_timeout(self, stub_registry: _StubRegistry) -> None:
        with pytest.raises(BashArgumentError):
            await bash_handler(
                "sess_01TEST",
                {"command": "true", "timeout_seconds": 0},
            )

    async def test_rejects_negative_timeout(self, stub_registry: _StubRegistry) -> None:
        with pytest.raises(BashArgumentError):
            await bash_handler(
                "sess_01TEST",
                {"command": "true", "timeout_seconds": -1},
            )


class TestResultShape:
    async def test_returns_structured_dict(
        self, stub_registry: _StubRegistry, stub_handle: SandboxHandle
    ) -> None:
        result = await bash_handler("sess_01TEST", {"command": "echo hello world"})
        assert result == {
            "exit_code": 0,
            "stdout": "hello world\n",
            "stderr": "",
            "timed_out": False,
            "truncated": False,
        }

    async def test_provisions_container_lazily(self, stub_registry: _StubRegistry) -> None:
        await bash_handler("sess_01TEST", {"command": "true"})
        assert stub_registry.get_or_provision_calls == ["sess_01TEST"]

    async def test_passes_command_to_container(
        self, stub_registry: _StubRegistry, stub_handle: SandboxHandle
    ) -> None:
        await bash_handler("sess_01TEST", {"command": "echo hi"})
        stub_registry.exec.assert_awaited_once()  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = stub_registry.exec.await_args.kwargs  # type: ignore[attr-defined]
        args: tuple[Any, ...] = stub_registry.exec.await_args.args  # type: ignore[attr-defined]
        # call signature is (handle, command, kwargs)
        assert args[1] == "echo hi"
        assert kwargs["timeout_seconds"] > 0
        assert kwargs["max_output_bytes"] > 0


class TestTimeoutCapping:
    async def test_caps_at_default_maximum(
        self,
        stub_registry: _StubRegistry,
        stub_handle: SandboxHandle,
    ) -> None:
        from aios.config import get_settings

        max_allowed = get_settings().bash_default_timeout_seconds
        await bash_handler(
            "sess_01TEST",
            {"command": "true", "timeout_seconds": max_allowed * 10},
        )
        kwargs: dict[str, Any] = stub_registry.exec.await_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["timeout_seconds"] == max_allowed

    async def test_respects_smaller_timeout(
        self,
        stub_registry: _StubRegistry,
        stub_handle: SandboxHandle,
    ) -> None:
        await bash_handler(
            "sess_01TEST",
            {"command": "true", "timeout_seconds": 5},
        )
        kwargs: dict[str, Any] = stub_registry.exec.await_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["timeout_seconds"] == 5


class TestExecRaisedReconcileSuppression:
    """When exec raises, reconcile exceptions are suppressed so the original propagates."""

    async def test_exec_exception_propagates_not_reconcile_exception(
        self, stub_registry: _StubRegistry
    ) -> None:
        """sandbox.exec raises; reconcile also raises; original exec exception propagates."""
        exec_error = RuntimeError("container died")
        reconcile_error = RuntimeError("reconcile failed too")

        stub_registry.exec.side_effect = exec_error  # type: ignore[attr-defined]

        with (
            patch(
                "aios.tools.bash.reconcile_memory_mounts",
                new_callable=AsyncMock,
                side_effect=reconcile_error,
            ) as mock_reconcile,
            pytest.raises(RuntimeError, match="container died"),
        ):
            await bash_handler("sess_01TEST", {"command": "true"})

        # reconcile was still attempted despite exec raising
        mock_reconcile.assert_awaited_once()
