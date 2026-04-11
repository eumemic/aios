"""Tests for the async-refactored ShellFileOperations.

This is the end-to-end validation for the Phase 4 async refactor. A
FakeTerminalEnv records every command sent to it and returns canned
ExecuteResults; all tests run the real vendored code paths against this
fake.

Any missed ``await`` in the refactored code produces a coroutine-as-value
bug that surfaces here as ``TypeError: 'coroutine' object is not
subscriptable`` or similar -- this is the safety net.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from aios.tools.adapters import ExecuteResult
from aios.vendor.hermes_files.file_operations import (
    PatchResult,
    ReadResult,
    SearchResult,
    ShellFileOperations,
    WriteResult,
)


class FakeTerminalEnv:
    """Records commands and returns canned or computed ExecuteResults.

    Usage: instantiate, then either queue substring-keyed responses via
    :meth:`queue` or attach a callable via :meth:`set_handler` to
    compute results based on the full command string. Any command not
    matched returns an empty-success ``ExecuteResult``.
    """

    def __init__(self) -> None:
        self.cwd = "/workspace"
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self._responses: list[tuple[str, ExecuteResult]] = []
        self._handler: Callable[[str], ExecuteResult | None] | None = None
        self._fallback = ExecuteResult(stdout="", exit_code=0)

    def queue(self, cmd_substring: str, result: ExecuteResult) -> None:
        """Queue a response for any command containing ``cmd_substring``."""
        self._responses.append((cmd_substring, result))

    def set_handler(self, fn: Callable[[str], ExecuteResult | None]) -> None:
        """Install a callable that computes responses on the fly."""
        self._handler = fn

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,  # noqa: ASYNC109  # mirrors hermes API
        stdin_data: str | None = None,
    ) -> ExecuteResult:
        self.commands.append((command, {"cwd": cwd, "timeout": timeout, "stdin_data": stdin_data}))
        if self._handler is not None:
            result = self._handler(command)
            if result is not None:
                return result
        for substring, result in self._responses:
            if substring in command:
                return result
        return self._fallback


@pytest.fixture
def env() -> FakeTerminalEnv:
    return FakeTerminalEnv()


@pytest.fixture
def ops(env: FakeTerminalEnv) -> ShellFileOperations:
    # Type-ignored because FakeTerminalEnv is duck-typed; the real
    # constructor expects SandboxTerminalEnv but any object with the
    # matching async run_command works.
    return ShellFileOperations(env)  # type: ignore[arg-type]


class TestExec:
    async def test_exec_passes_through(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        env.queue("echo hello", ExecuteResult(stdout="hello\n", exit_code=0))
        result = await ops._exec("echo hello")
        assert result.stdout == "hello\n"
        assert result.exit_code == 0
        assert env.commands[0][0] == "echo hello"
        assert env.commands[0][1]["cwd"] == "/workspace"

    async def test_exec_custom_cwd(self, env: FakeTerminalEnv, ops: ShellFileOperations) -> None:
        await ops._exec("pwd", cwd="/tmp")
        assert env.commands[0][1]["cwd"] == "/tmp"


class TestHasCommand:
    async def test_has_command_caches(self, env: FakeTerminalEnv, ops: ShellFileOperations) -> None:
        env.queue("command -v rg", ExecuteResult(stdout="yes\n", exit_code=0))
        assert await ops._has_command("rg") is True
        assert await ops._has_command("rg") is True
        rg_commands = [c for c, _ in env.commands if "command -v rg" in c]
        assert len(rg_commands) == 1

    async def test_has_command_missing(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        env.queue("command -v nosuch", ExecuteResult(stdout="", exit_code=1))
        assert await ops._has_command("nosuch") is False


class TestPureHelpers:
    def test_add_line_numbers(self, ops: ShellFileOperations) -> None:
        numbered = ops._add_line_numbers("hello\nworld", start_line=10)
        lines = numbered.split("\n")
        assert lines[0].endswith("|hello")
        assert lines[1].endswith("|world")
        assert "    10|" in lines[0]
        assert "    11|" in lines[1]

    def test_add_line_numbers_truncates_long_lines(self, ops: ShellFileOperations) -> None:
        long = "x" * 3000
        numbered = ops._add_line_numbers(long)
        assert "[truncated]" in numbered

    def test_escape_shell_arg(self, ops: ShellFileOperations) -> None:
        assert ops._escape_shell_arg("hello") == "'hello'"
        assert "'\"'\"'" in ops._escape_shell_arg("it's")

    def test_is_likely_binary_by_extension(self, ops: ShellFileOperations) -> None:
        assert ops._is_likely_binary("/workspace/foo.png") is True
        assert ops._is_likely_binary("/workspace/foo.txt") is False

    def test_is_likely_binary_by_content(self, ops: ShellFileOperations) -> None:
        assert ops._is_likely_binary("/workspace/foo.txt", content_sample="hello") is False
        assert (
            ops._is_likely_binary(
                "/workspace/foo.txt",
                content_sample="\x00\x01\x02\x03" * 100,
            )
            is True
        )

    def test_unified_diff(self, ops: ShellFileOperations) -> None:
        diff = ops._unified_diff("line1\nline2\n", "line1\nline2changed\n", "x.py")
        assert "a/x.py" in diff
        assert "b/x.py" in diff
        assert "-line2" in diff
        assert "+line2changed" in diff


class TestReadFile:
    async def test_read_file_happy_path(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        def handler(cmd: str) -> ExecuteResult | None:
            if cmd == "echo $HOME":
                return ExecuteResult(stdout="/root\n", exit_code=0)
            if "wc -c" in cmd:
                return ExecuteResult(stdout="12\n", exit_code=0)
            if "head -c 1000" in cmd:
                return ExecuteResult(stdout="hello world\n", exit_code=0)
            if "sed -n" in cmd:
                return ExecuteResult(stdout="hello world\n", exit_code=0)
            if "wc -l" in cmd:
                return ExecuteResult(stdout="1\n", exit_code=0)
            return None

        env.set_handler(handler)
        result = await ops.read_file("/workspace/hello.txt")
        assert isinstance(result, ReadResult)
        assert result.error is None
        assert "hello world" in result.content
        assert "     1|hello world" in result.content
        assert result.total_lines == 1
        assert result.file_size == 12

    async def test_read_file_binary_rejected(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        def handler(cmd: str) -> ExecuteResult | None:
            if "wc -c" in cmd:
                return ExecuteResult(stdout="1024\n", exit_code=0)
            if "head -c 1000" in cmd:
                return ExecuteResult(stdout="\x00\x01\x02" * 100, exit_code=0)
            return None

        env.set_handler(handler)
        result = await ops.read_file("/workspace/image.png")
        assert result.is_binary is True
        assert result.error is not None
        assert "Binary file" in result.error

    async def test_read_file_not_found_suggests_similar(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        def handler(cmd: str) -> ExecuteResult | None:
            if "wc -c" in cmd:
                return ExecuteResult(stdout="", exit_code=1)
            if "ls -1" in cmd:
                return ExecuteResult(stdout="hello.txt\nhello_world.txt\n", exit_code=0)
            return None

        env.set_handler(handler)
        result = await ops.read_file("/workspace/hellox.txt")
        assert result.error is not None
        assert "not found" in result.error.lower()
        assert len(result.similar_files) > 0


class TestWriteFile:
    async def test_write_file_happy_path(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        def handler(cmd: str) -> ExecuteResult | None:
            if "mkdir -p" in cmd:
                return ExecuteResult(stdout="", exit_code=0)
            if "cat >" in cmd:
                return ExecuteResult(stdout="", exit_code=0)
            if "wc -c" in cmd:
                return ExecuteResult(stdout="5\n", exit_code=0)
            return None

        env.set_handler(handler)
        result = await ops.write_file("/workspace/subdir/hi.txt", "hello")
        assert isinstance(result, WriteResult)
        assert result.error is None
        assert result.bytes_written == 5
        assert result.dirs_created is True

        # Verify the write command got the content via stdin_data, NOT
        # embedded in the command string -- this is the ARG_MAX defense.
        write_cmds = [(cmd, kw) for cmd, kw in env.commands if "cat >" in cmd]
        assert len(write_cmds) == 1
        assert write_cmds[0][1]["stdin_data"] == "hello"

    async def test_write_denied_path_returns_error(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        # The deny list includes /etc/passwd regardless of host config.
        result = await ops.write_file("/etc/passwd", "evil")
        assert result.error is not None
        assert "denied" in result.error.lower()
        # No commands should have run -- denial happens before any exec.
        assert len(env.commands) == 0


class TestPatchReplace:
    async def test_patch_replace_happy_path(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        state = {"content": "def foo():\n    pass\n"}

        def handler(cmd: str) -> ExecuteResult | None:
            # The read is `cat '/path' 2>/dev/null` — match on the read shape.
            if cmd.startswith("cat '") and "2>/dev/null" in cmd:
                return ExecuteResult(stdout=state["content"], exit_code=0)
            if "mkdir -p" in cmd:
                return ExecuteResult(stdout="", exit_code=0)
            # The write is `cat > '/path'` — match on the > operator.
            if "cat > " in cmd:
                return ExecuteResult(stdout="", exit_code=0)
            if "wc -c" in cmd:
                return ExecuteResult(stdout="25\n", exit_code=0)
            return None

        env.set_handler(handler)
        result = await ops.patch_replace("/workspace/f.py", "def foo():", "def bar():")
        assert isinstance(result, PatchResult)
        assert result.success is True
        assert "def bar" in result.diff or "+def bar" in result.diff
        assert result.files_modified == ["/workspace/f.py"]

    async def test_patch_replace_no_match_returns_error(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        def handler(cmd: str) -> ExecuteResult | None:
            if cmd.startswith("cat '") and "2>/dev/null" in cmd:
                return ExecuteResult(stdout="def other():\n    pass\n", exit_code=0)
            return None

        env.set_handler(handler)
        result = await ops.patch_replace("/workspace/f.py", "def foo():", "def bar():")
        assert result.success is False
        assert result.error is not None
        assert "match" in result.error.lower() or "find" in result.error.lower()


class TestSearch:
    async def test_search_files_mode_uses_rg(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        def handler(cmd: str) -> ExecuteResult | None:
            if "command -v rg" in cmd:
                return ExecuteResult(stdout="yes\n", exit_code=0)
            if "test -e" in cmd:
                return ExecuteResult(stdout="exists\n", exit_code=0)
            if "rg --files" in cmd:
                return ExecuteResult(stdout="/workspace/a.py\n/workspace/b.py\n", exit_code=0)
            return None

        env.set_handler(handler)
        result = await ops.search("*.py", target="files")
        assert isinstance(result, SearchResult)
        assert result.files == ["/workspace/a.py", "/workspace/b.py"]
        assert result.total_count == 2

    async def test_search_content_with_rg(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        def handler(cmd: str) -> ExecuteResult | None:
            if "command -v rg" in cmd:
                return ExecuteResult(stdout="yes\n", exit_code=0)
            if "test -e" in cmd:
                return ExecuteResult(stdout="exists\n", exit_code=0)
            if "--line-number" in cmd:
                return ExecuteResult(
                    stdout="/workspace/a.py:1:def foo():\n/workspace/b.py:3:def bar():\n",
                    exit_code=0,
                )
            return None

        env.set_handler(handler)
        result = await ops.search(r"def \w+\(", target="content")
        assert len(result.matches) == 2
        assert result.matches[0].path == "/workspace/a.py"
        assert result.matches[0].line_number == 1
        assert "def foo" in result.matches[0].content

    async def test_search_path_not_found(
        self, env: FakeTerminalEnv, ops: ShellFileOperations
    ) -> None:
        def handler(cmd: str) -> ExecuteResult | None:
            if "test -e" in cmd:
                return ExecuteResult(stdout="not_found\n", exit_code=0)
            return None

        env.set_handler(handler)
        result = await ops.search("x", path="/nonexistent")
        assert result.error is not None
        assert "not found" in result.error.lower()


class TestWriteDenyPureFunction:
    """Pure tests of _is_write_denied without touching the env."""

    def test_ssh_authorized_keys_denied(self) -> None:
        from pathlib import Path

        from aios.vendor.hermes_files.file_operations import _is_write_denied

        home = Path.home()
        assert _is_write_denied(str(home / ".ssh" / "authorized_keys")) is True

    def test_random_workspace_file_allowed(self) -> None:
        from aios.vendor.hermes_files.file_operations import _is_write_denied

        assert _is_write_denied("/workspace/foo.txt") is False
        assert _is_write_denied("/tmp/bar") is False

    def test_etc_passwd_denied(self) -> None:
        from aios.vendor.hermes_files.file_operations import _is_write_denied

        assert _is_write_denied("/etc/passwd") is True
