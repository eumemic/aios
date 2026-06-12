"""Unit tests for the bash tool handler (with mocked container).

Exercises argument validation and result shaping. The sandbox registry
is stubbed so tests don't touch Docker.
"""

from __future__ import annotations

from collections.abc import Generator
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
        owner_id="sess_01TEST",
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
) -> Generator[_StubRegistry]:
    """Install a stub sandbox registry on the runtime module, restore after.

    Also pins ``resolve_bash_timeout_ceiling`` to the global default so the
    handler's per-env timeout lookup (issue #725) doesn't try to read a
    real session row off the MagicMock pool. Tests that exercise the
    per-env ceiling patch this themselves (see ``TestPerEnvTimeoutCeiling``).
    """
    from unittest.mock import MagicMock

    from aios.config import get_settings

    prev_registry = runtime.sandbox_registry
    prev_pool = runtime.pool
    stub = _StubRegistry(stub_handle, canned_result)
    runtime.sandbox_registry = stub  # type: ignore[assignment]
    runtime.pool = MagicMock()
    default_ceiling = get_settings().bash_default_timeout_seconds
    with patch(
        "aios.tools.bash.resolve_bash_timeout_ceiling",
        new_callable=AsyncMock,
        return_value=default_ceiling,
    ):
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
        stub_registry.exec.assert_awaited_once()
        assert stub_registry.exec.await_args is not None
        kwargs: dict[str, Any] = dict(stub_registry.exec.await_args.kwargs)
        args: tuple[Any, ...] = stub_registry.exec.await_args.args
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
        assert stub_registry.exec.await_args is not None
        kwargs: dict[str, Any] = dict(stub_registry.exec.await_args.kwargs)
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
        assert stub_registry.exec.await_args is not None
        kwargs: dict[str, Any] = dict(stub_registry.exec.await_args.kwargs)
        assert kwargs["timeout_seconds"] == 5


class TestPerEnvTimeoutCeiling:
    """The handler clamps to the per-environment ceiling resolved by
    ``resolve_bash_timeout_ceiling`` (issue #725), not the global default.

    The fixture pins the resolver to the global default; these tests
    override it to a higher (or lower) per-env value to prove the handler
    honors whatever ceiling the resolver returns.
    """

    async def test_honors_higher_per_env_ceiling(self, stub_registry: _StubRegistry) -> None:
        """A request between the global default and a higher per-env
        ceiling is passed through (not clamped to the global default)."""
        with patch(
            "aios.tools.bash.resolve_bash_timeout_ceiling",
            new_callable=AsyncMock,
            return_value=600,
        ):
            await bash_handler(
                "sess_01TEST",
                {"command": "true", "timeout_seconds": 300},
            )
        assert stub_registry.exec.await_args is not None
        kwargs: dict[str, Any] = dict(stub_registry.exec.await_args.kwargs)
        # 300 < 600 ceiling → request honored verbatim.
        assert kwargs["timeout_seconds"] == 300

    async def test_caps_at_higher_per_env_ceiling(self, stub_registry: _StubRegistry) -> None:
        """A request above the per-env ceiling is clamped to that ceiling."""
        with patch(
            "aios.tools.bash.resolve_bash_timeout_ceiling",
            new_callable=AsyncMock,
            return_value=600,
        ):
            await bash_handler(
                "sess_01TEST",
                {"command": "true", "timeout_seconds": 5000},
            )
        assert stub_registry.exec.await_args is not None
        kwargs: dict[str, Any] = dict(stub_registry.exec.await_args.kwargs)
        assert kwargs["timeout_seconds"] == 600

    async def test_default_to_per_env_ceiling_when_unspecified(
        self, stub_registry: _StubRegistry
    ) -> None:
        """With no ``timeout_seconds`` in the call, the handler defaults to
        the resolved per-env ceiling (here above the global default)."""
        with patch(
            "aios.tools.bash.resolve_bash_timeout_ceiling",
            new_callable=AsyncMock,
            return_value=600,
        ):
            await bash_handler("sess_01TEST", {"command": "true"})
        assert stub_registry.exec.await_args is not None
        kwargs: dict[str, Any] = dict(stub_registry.exec.await_args.kwargs)
        assert kwargs["timeout_seconds"] == 600


class TestExecRaisedReconcileSuppression:
    """When exec raises, reconcile exceptions are suppressed so the original propagates."""

    async def test_exec_exception_propagates_not_reconcile_exception(
        self, stub_registry: _StubRegistry
    ) -> None:
        """sandbox.exec raises; reconcile also raises; original exec exception propagates."""
        exec_error = RuntimeError("container died")
        reconcile_error = RuntimeError("reconcile failed too")

        stub_registry.exec.side_effect = exec_error

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
