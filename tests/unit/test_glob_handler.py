"""Unit tests for the glob tool handler.

Stubs the sandbox registry and mocks ContainerHandle.run_command so the
tests don't touch Docker. The same pattern as test_read_handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.sandbox.container import CommandResult, ContainerHandle
from aios.tools.glob import GlobArgumentError, glob_handler


class _StubRegistry:
    def __init__(self, handle: ContainerHandle) -> None:
        self._handle = handle

    async def get_or_provision(self, session_id: str, **_kwargs: Any) -> ContainerHandle:
        return self._handle


@pytest.fixture
def stub_handle() -> ContainerHandle:
    handle = ContainerHandle(
        session_id="sess_01TEST",
        container_id="container_abc",
        workspace_path=Path("/tmp/aios-test"),
    )
    handle.run_command = AsyncMock(  # type: ignore[method-assign]
        return_value=CommandResult(
            exit_code=0,
            stdout="/workspace/foo.py\n/workspace/bar.py\n",
            stderr="",
            timed_out=False,
            truncated=False,
        )
    )
    return handle


@pytest.fixture
def stub_registry(stub_handle: ContainerHandle) -> Any:
    from unittest.mock import MagicMock

    prev_registry = runtime.sandbox_registry
    prev_pool = runtime.pool
    runtime.sandbox_registry = _StubRegistry(stub_handle)  # type: ignore[assignment]
    runtime.pool = MagicMock()
    try:
        yield
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
    async def test_returns_matches(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        result = await glob_handler("sess_01TEST", {"pattern": "*.py"})
        assert result == {"matches": ["/workspace/foo.py", "/workspace/bar.py"]}

    async def test_default_path(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await glob_handler("sess_01TEST", {"pattern": "*.py"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "/workspace" in cmd
        assert "rg --files" in cmd
        assert "*.py" in cmd

    async def test_custom_path(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await glob_handler("sess_01TEST", {"pattern": "*.txt", "path": "/workspace/docs"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "/workspace/docs" in cmd

    async def test_empty_results(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        stub_handle.run_command = AsyncMock(  # type: ignore[method-assign]
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
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=2,
                stdout="",
                stderr="find: some error\n",
                timed_out=False,
                truncated=False,
            )
        )
        result = await glob_handler("sess_01TEST", {"pattern": "*.py"})
        assert "error" in result
        assert "some error" in result["error"]
