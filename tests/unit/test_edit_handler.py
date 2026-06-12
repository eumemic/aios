"""Unit tests for the edit tool handler.

Mocks SandboxHandle.run_command to serve canned ``cat`` results on
the first call and swallow the base64 write-back on the second call.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.harness import runtime
from aios.sandbox.backends.base import CommandResult, SandboxHandle
from aios.tools.edit import EditArgumentError, edit_handler


class _StubRegistry:
    """Minimal stand-in for SandboxRegistry used by handler tests."""

    def __init__(self, handle: SandboxHandle, result: CommandResult) -> None:
        self._handle = handle
        self.exec = AsyncMock(return_value=result)

    async def get_or_provision(self, session_id: str, **_kwargs: Any) -> SandboxHandle:
        return self._handle


def _ok(stdout: str = "") -> CommandResult:
    return CommandResult(exit_code=0, stdout=stdout, stderr="", timed_out=False, truncated=False)


def _err(exit_code: int, stderr: str) -> CommandResult:
    return CommandResult(
        exit_code=exit_code, stdout="", stderr=stderr, timed_out=False, truncated=False
    )


@pytest.fixture
def stub_handle(**kwargs: Any) -> SandboxHandle:
    return SandboxHandle(
        owner_id="sess_01TEST",
        sandbox_id="container_abc",
        workspace_path=Path("/tmp/aios-test"),
    )


@pytest.fixture
def stub_registry(stub_handle: SandboxHandle, **kwargs: Any) -> Any:
    from unittest.mock import MagicMock

    prev_registry = runtime.sandbox_registry
    prev_pool = runtime.pool
    stub = _StubRegistry(stub_handle, _ok())
    runtime.sandbox_registry = stub  # type: ignore[assignment]
    runtime.pool = MagicMock()
    try:
        yield stub
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
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = _script_responses()
        result = await edit_handler(
            "sess_01TEST",
            {"path": "/workspace/a.txt", "old_string": "a", "new_string": "a"},
        )
        assert "error" in result
        assert "identical" in result["error"]
        # Container was never touched — guard short-circuits before cat.
        stub_registry.exec.assert_not_awaited()


class TestStrictMatching:
    async def test_not_found_returns_error_with_retry_hint(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = _script_responses(_ok("actual file content\n"))
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
        assert stub_registry.exec.await_count == 1

    async def test_multiple_matches_requires_replace_all(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = _script_responses(_ok("foo\nbar\nfoo\nfoo\n"))
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
        assert stub_registry.exec.await_count == 1

    async def test_unique_match_replaces_and_writes_back(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = _script_responses(
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
        assert stub_registry.exec.await_count == 2

    async def test_write_back_uses_modified_content(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = _script_responses(
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
        write_cmd: str = stub_registry.exec.await_args_list[1].args[1]
        expected_b64 = base64.b64encode(b"alpha BETA gamma\n").decode("ascii")
        assert expected_b64 in write_cmd
        # shlex.quote leaves simple paths unquoted.
        assert "> /workspace/a.txt" in write_cmd

    async def test_replace_all_replaces_every_occurrence(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = _script_responses(
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
        write_cmd: str = stub_registry.exec.await_args_list[1].args[1]
        expected_b64 = base64.b64encode(b"FOO\nbar\nFOO\nFOO\n").decode("ascii")
        assert expected_b64 in write_cmd


class TestPerEnvTimeoutCeiling:
    """edit routes BOTH its read-back and write-back sandbox execs through
    the per-environment bash-timeout ceiling (#725), not the hardcoded
    global default."""

    async def test_both_execs_use_resolved_ceiling(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = _script_responses(
            _ok("alpha beta gamma\n"),
            _ok(""),
        )
        with patch(
            "aios.tools.edit.resolve_bash_timeout_ceiling",
            new_callable=AsyncMock,
            return_value=600,
        ):
            await edit_handler(
                "sess_01TEST",
                {"path": "/workspace/a.txt", "old_string": "beta", "new_string": "BETA"},
            )
        for call in stub_registry.exec.await_args_list:
            assert call.kwargs["timeout_seconds"] == 600


class TestErrorPaths:
    async def test_cat_failure_returns_error_dict(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = _script_responses(_err(1, "cat: /nope: No such file or directory\n"))
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
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = _script_responses(
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

    async def test_truncated_cat_aborts_write_back(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        """``cat -- path`` is capped at ``bash_max_output_bytes`` (100 KiB by
        default). For a non-memory file larger than that, the docker backend
        sets ``CommandResult.truncated=True`` and appends the literal
        ``"\\n\\n[output truncated]"`` marker to ``stdout``. Edit previously
        ignored the flag, treated the truncated bytes as the full file, ran
        ``original.replace(old, new)``, and write-back replaced the entire
        file with the truncated content (plus the marker text) — silent
        data loss for every model edit on any non-trivial file.

        Failing pre-fix because edit accepted the truncated read and wrote
        back; post-fix because edit now returns a typed error referencing
        truncation, never reaching the second exec.
        """
        truncated_stdout = "A" * 95000 + "TARGET\n" + "rest..." + "\n\n[output truncated]"
        truncated_result = CommandResult(
            exit_code=0,
            stdout=truncated_stdout,
            stderr="",
            timed_out=False,
            truncated=True,
        )
        stub_registry.exec = _script_responses(truncated_result, _ok(""))
        result = await edit_handler(
            "sess_01TEST",
            {
                "path": "/workspace/big.txt",
                "old_string": "TARGET",
                "new_string": "REPLACED",
            },
        )
        assert "error" in result, (
            f"edit must refuse to write back when the read was truncated "
            f"(otherwise the bytes past the truncation point are silently "
            f"lost on write); got success {result!r}."
        )
        assert "truncat" in result["error"].lower(), (
            f"error must reference truncation so the model can use a "
            f"different tool (e.g., read with explicit ranges); got "
            f"{result['error']!r}."
        )
        # Critically, no write-back was attempted.
        assert stub_registry.exec.await_count == 1
