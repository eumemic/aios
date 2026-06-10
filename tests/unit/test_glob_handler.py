"""Unit tests for the glob tool handler.

Stubs the sandbox registry and mocks SandboxHandle.run_command so the
tests don't touch Docker. The same pattern as test_read_handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.harness import runtime
from aios.sandbox.backends.base import CommandResult, SandboxHandle
from aios.tools.glob import GlobArgumentError, glob_handler


class _StubRegistry:
    """Minimal stand-in for SandboxRegistry used by handler tests."""

    def __init__(self, handle: SandboxHandle, result: CommandResult) -> None:
        self._handle = handle
        self.exec = AsyncMock(return_value=result)

    async def get_or_provision(self, session_id: str, **_kwargs: Any) -> SandboxHandle:
        return self._handle


@pytest.fixture
def stub_handle(**kwargs: Any) -> SandboxHandle:
    handle = SandboxHandle(
        session_id="sess_01TEST",
        sandbox_id="container_abc",
        workspace_path=Path("/tmp/aios-test"),
    )
    return handle


@pytest.fixture
def canned_result() -> CommandResult:
    return CommandResult(
        exit_code=0,
        stdout="/workspace/foo.py\n/workspace/bar.py\n",
        stderr="",
        timed_out=False,
        truncated=False,
    )


@pytest.fixture
def stub_registry(stub_handle: SandboxHandle, canned_result: CommandResult, **kwargs: Any) -> Any:
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
    async def test_rejects_missing_pattern(self, stub_registry: Any) -> None:
        with pytest.raises(GlobArgumentError):
            await glob_handler("sess_01TEST", {})

    async def test_rejects_empty_pattern(self, stub_registry: Any) -> None:
        with pytest.raises(GlobArgumentError):
            await glob_handler("sess_01TEST", {"pattern": "   "})

    async def test_rejects_non_string_pattern(self, stub_registry: Any) -> None:
        with pytest.raises(GlobArgumentError):
            await glob_handler("sess_01TEST", {"pattern": 42})

    async def test_rejects_empty_path(self, stub_registry: Any) -> None:
        with pytest.raises(GlobArgumentError):
            await glob_handler("sess_01TEST", {"pattern": "*.py", "path": "   "})


class TestHappyPath:
    async def test_returns_matches(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        result = await glob_handler("sess_01TEST", {"pattern": "*.py"})
        assert result == {"matches": ["/workspace/foo.py", "/workspace/bar.py"]}

    async def test_default_path(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await glob_handler("sess_01TEST", {"pattern": "*.py"})
        cmd: str = stub_registry.exec.await_args.args[1]
        assert "/workspace" in cmd
        assert "rg --files" in cmd
        assert "*.py" in cmd

    async def test_custom_path(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await glob_handler("sess_01TEST", {"pattern": "*.txt", "path": "/workspace/docs"})
        cmd: str = stub_registry.exec.await_args.args[1]
        assert "/workspace/docs" in cmd

    async def test_empty_results(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        stub_registry.exec = AsyncMock(
            return_value=CommandResult(
                exit_code=0,
                stdout="",
                stderr="",
                timed_out=False,
                truncated=False,
            )
        )
        result = await glob_handler("sess_01TEST", {"pattern": "*.xyz"})
        assert result == {"matches": []}


class TestErrorPath:
    async def test_find_failure_returns_error_dict(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = AsyncMock(
            return_value=CommandResult(
                exit_code=2,
                stdout="",
                stderr="find: some error\\n",
                timed_out=False,
                truncated=False,
            )
        )
        result = await glob_handler("sess_01TEST", {"pattern": "*.py"})
        assert "error" in result
        assert "some error" in result["error"]

    async def test_cmd_uses_pipefail_so_rg_failure_propagates(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        """Same shape as the grep pipefail fix (PR #546): ``rg --files
        --glob <pattern> <path> 2>/dev/null | head -500`` is a pipe
        whose final exit code is ``head``'s — which is 0 even when
        ``rg`` failed (invalid glob pattern, missing path), because
        ``head`` happily consumes empty input and exits 0.  Without
        ``set -o pipefail`` the result returned to the model is
        ``{"matches": []}`` — looking like "no matches" rather than
        the actual rg error.  The model chases red herrings: tries
        different patterns, different paths, never realizing rg
        itself rejected its input.

        PR #546's commit body named this as the next-up sibling
        ("Sibling bug exists in tools/glob.py").
        """
        await glob_handler("sess_01TEST", {"pattern": "*.py"})
        cmd: str = stub_registry.exec.await_args.args[1]
        assert "pipefail" in cmd, (
            f"cmd must enable ``pipefail`` so a failing ``rg`` (e.g., "
            f"invalid glob pattern, missing path) propagates through the "
            f"``| head -N`` to the overall pipe exit code; without it, "
            f"rg errors silently return empty matches. Got: {cmd!r}"
        )


class TestPerEnvTimeoutCeiling:
    """glob routes its sandbox exec through the per-environment bash-timeout
    ceiling (#725), not the hardcoded global default."""

    async def test_exec_uses_resolved_ceiling(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        with patch(
            "aios.tools.glob.resolve_bash_timeout_ceiling",
            new_callable=AsyncMock,
            return_value=600,
        ):
            await glob_handler("sess_01TEST", {"pattern": "*.py"})
        kwargs: dict[str, Any] = stub_registry.exec.await_args.kwargs
        assert kwargs["timeout_seconds"] == 600
