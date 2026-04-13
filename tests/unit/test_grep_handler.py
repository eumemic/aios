"""Unit tests for the grep tool handler.

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
from aios.tools.grep import GrepArgumentError, grep_handler


class _StubRegistry:
    def __init__(self, handle: ContainerHandle) -> None:
        self._handle = handle

    async def get_or_provision(self, session_id: str) -> ContainerHandle:
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
            stdout="/workspace/foo.py:10:def hello():\n/workspace/bar.py:3:import os\n",
            stderr="",
            timed_out=False,
            truncated=False,
        )
    )
    return handle


@pytest.fixture
def stub_registry(stub_handle: ContainerHandle) -> Any:
    previous = runtime.sandbox_registry
    runtime.sandbox_registry = _StubRegistry(stub_handle)  # type: ignore[assignment]
    try:
        yield
    finally:
        runtime.sandbox_registry = previous


class TestArguments:
    async def test_rejects_missing_pattern(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {})

    async def test_rejects_empty_pattern(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {"pattern": "   "})

    async def test_rejects_non_string_pattern(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {"pattern": 42})

    async def test_rejects_empty_path(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {"pattern": "hello", "path": "   "})

    async def test_rejects_empty_include(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {"pattern": "hello", "include": "   "})


class TestHappyPath:
    async def test_returns_matches(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        result = await grep_handler("sess_01TEST", {"pattern": "hello"})
        assert result == {
            "matches": "/workspace/foo.py:10:def hello():\n/workspace/bar.py:3:import os\n",
        }

    async def test_default_path(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "grep -rn" in cmd
        assert "/workspace" in cmd

    async def test_include_flag(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "include": "*.py"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "--include=" in cmd
        assert "*.py" in cmd

    async def test_no_matches_returns_empty_string(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=1,
                stdout="",
                stderr="",
                timed_out=False,
                truncated=False,
            )
        )
        result = await grep_handler("sess_01TEST", {"pattern": "nonexistent"})
        assert result == {"matches": ""}


class TestErrorPath:
    async def test_grep_failure_returns_error_dict(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=2,
                stdout="",
                stderr="grep: invalid regex\n",
                timed_out=False,
                truncated=False,
            )
        )
        result = await grep_handler("sess_01TEST", {"pattern": "[invalid"})
        assert "error" in result
        assert "invalid regex" in result["error"]
