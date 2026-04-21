"""Unit tests for the edit tool handler.

Mocks ContainerHandle.run_command to serve canned ``cat`` results on
the first call and swallow the base64 write-back on the second call.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.sandbox.container import CommandResult, ContainerHandle
from aios.tools.edit import EditArgumentError, edit_handler


class _StubRegistry:
    def __init__(self, handle: ContainerHandle) -> None:
        self._handle = handle

    async def get_or_provision(self, session_id: str, **_kwargs: Any) -> ContainerHandle:
        return self._handle


def _ok(stdout: str = "") -> CommandResult:
    return CommandResult(exit_code=0, stdout=stdout, stderr="", timed_out=False, truncated=False)


def _err(exit_code: int, stderr: str) -> CommandResult:
    return CommandResult(
        exit_code=exit_code, stdout="", stderr=stderr, timed_out=False, truncated=False
    )


@pytest.fixture
def stub_handle() -> ContainerHandle:
    return ContainerHandle(
        session_id="sess_01TEST",
        container_id="container_abc",
        workspace_path=Path("/tmp/aios-test"),
    )


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


def _script_responses(*responses: CommandResult) -> AsyncMock:
    """Return an AsyncMock that serves each response in order per call."""
    return AsyncMock(side_effect=list(responses))


class TestArguments:
    async def test_rejects_missing_path(self, stub_registry: Any) -> None:
        with pytest.raises(EditArgumentError):
            await edit_handler("sess_01TEST", {"old_string": "a", "new_string": "b"})

    async def test_rejects_empty_old_string(self, stub_registry: Any) -> None:
        with pytest.raises(EditArgumentError):
            await edit_handler(
                "sess_01TEST",
                {"path": "/workspace/a.txt", "old_string": "", "new_string": "b"},
            )

    async def test_rejects_missing_new_string(self, stub_registry: Any) -> None:
        with pytest.raises(EditArgumentError):
            await edit_handler(
                "sess_01TEST",
                {"path": "/workspace/a.txt", "old_string": "a"},
            )


class TestIdenticalStrings:
    async def test_identical_old_and_new_returns_error(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = _script_responses()  # type: ignore[method-assign]
        result = await edit_handler(
            "sess_01TEST",
            {"path": "/workspace/a.txt", "old_string": "a", "new_string": "a"},
        )
        assert "error" in result
        assert "identical" in result["error"]
        # Container was never touched — guard short-circuits before cat.
        stub_handle.run_command.assert_not_awaited()  # type: ignore[attr-defined]


class TestStrictMatching:
    async def test_not_found_returns_error_with_retry_hint(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = _script_responses(_ok("actual file content\n"))  # type: ignore[method-assign]
        result = await edit_handler(
            "sess_01TEST",
            {
                "path": "/workspace/a.txt",
                "old_string": "missing",
                "new_string": "replacement",
            },
        )
        assert "error" in result
        assert "not found" in result["error"]
        assert "read tool" in result["error"]
        # Only the read happened; no write-back because we errored out.
        assert stub_handle.run_command.await_count == 1  # type: ignore[attr-defined]

    async def test_multiple_matches_requires_replace_all(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = _script_responses(_ok("foo\nbar\nfoo\nfoo\n"))  # type: ignore[method-assign]
        result = await edit_handler(
            "sess_01TEST",
            {
                "path": "/workspace/a.txt",
                "old_string": "foo",
                "new_string": "baz",
            },
        )
        assert "error" in result
        assert result["matches"] == 3
        assert stub_handle.run_command.await_count == 1  # type: ignore[attr-defined]

    async def test_unique_match_replaces_and_writes_back(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = _script_responses(  # type: ignore[method-assign]
            _ok("hello world\n"),  # cat response
            _ok(""),  # write-back response
        )
        result = await edit_handler(
            "sess_01TEST",
            {
                "path": "/workspace/a.txt",
                "old_string": "hello",
                "new_string": "goodbye",
            },
        )
        assert "error" not in result
        assert result["path"] == "/workspace/a.txt"
        assert result["replaced"] == 1
        # Diff shows the change
        assert "-hello world" in result["diff"]
        assert "+goodbye world" in result["diff"]
        # Two run_command calls: cat and base64 write-back
        assert stub_handle.run_command.await_count == 2  # type: ignore[attr-defined]

    async def test_write_back_uses_modified_content(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = _script_responses(  # type: ignore[method-assign]
            _ok("alpha beta gamma\n"),
            _ok(""),
        )
        await edit_handler(
            "sess_01TEST",
            {
                "path": "/workspace/a.txt",
                "old_string": "beta",
                "new_string": "BETA",
            },
        )
        write_cmd: str = stub_handle.run_command.await_args_list[1].args[0]  # type: ignore[attr-defined]
        expected_b64 = base64.b64encode(b"alpha BETA gamma\n").decode("ascii")
        assert expected_b64 in write_cmd
        # shlex.quote leaves simple paths unquoted.
        assert "> /workspace/a.txt" in write_cmd

    async def test_replace_all_replaces_every_occurrence(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = _script_responses(  # type: ignore[method-assign]
            _ok("foo\nbar\nfoo\nfoo\n"),
            _ok(""),
        )
        result = await edit_handler(
            "sess_01TEST",
            {
                "path": "/workspace/a.txt",
                "old_string": "foo",
                "new_string": "FOO",
                "replace_all": True,
            },
        )
        assert "error" not in result
        assert result["replaced"] == 3
        write_cmd: str = stub_handle.run_command.await_args_list[1].args[0]  # type: ignore[attr-defined]
        expected_b64 = base64.b64encode(b"FOO\nbar\nFOO\nFOO\n").decode("ascii")
        assert expected_b64 in write_cmd


class TestErrorPaths:
    async def test_cat_failure_returns_error_dict(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = _script_responses(  # type: ignore[method-assign]
            _err(1, "cat: /nope: No such file or directory\n")
        )
        result = await edit_handler(
            "sess_01TEST",
            {
                "path": "/nope",
                "old_string": "hi",
                "new_string": "bye",
            },
        )
        assert "error" in result
        assert "No such file" in result["error"]

    async def test_write_back_failure_returns_error_dict(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = _script_responses(  # type: ignore[method-assign]
            _ok("hello\n"),
            _err(1, "bash: /ro/a.txt: Permission denied\n"),
        )
        result = await edit_handler(
            "sess_01TEST",
            {
                "path": "/ro/a.txt",
                "old_string": "hello",
                "new_string": "goodbye",
            },
        )
        assert "error" in result
        assert "Permission denied" in result["error"]
