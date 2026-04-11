"""Unit tests for the edit tool handler.

Stubs the sandbox registry and injects a test ShellFileOperations so
the tests don't touch Docker.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.tools import file_session
from aios.tools.edit import EditArgumentError, edit_handler
from aios.vendor.hermes_files.file_operations import PatchResult


class _StubFileOps:
    def __init__(self) -> None:
        self.replace_calls: list[tuple[str, str, str, bool]] = []
        self.patch_v4a_calls: list[str] = []
        self._replace_result = PatchResult(
            success=True,
            diff="--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-old\n+new\n",
            files_modified=["/workspace/a.py"],
        )
        self._patch_v4a_result = PatchResult(
            success=True,
            diff="...",
            files_modified=["/workspace/a.py"],
        )
        self._mtime: float | None = 1000.0

    def set_replace_result(self, result: PatchResult) -> None:
        self._replace_result = result

    def set_patch_v4a_result(self, result: PatchResult) -> None:
        self._patch_v4a_result = result

    async def patch_replace(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> PatchResult:
        self.replace_calls.append((path, old_string, new_string, replace_all))
        return self._replace_result

    async def patch_v4a(self, patch_content: str) -> PatchResult:
        self.patch_v4a_calls.append(patch_content)
        return self._patch_v4a_result

    async def _exec(self, command: str, **kwargs: Any) -> Any:
        class _R:
            def __init__(self, stdout: str, exit_code: int) -> None:
                self.stdout = stdout
                self.exit_code = exit_code

        if "stat -c %Y" in command:
            if self._mtime is None:
                return _R(stdout="", exit_code=1)
            return _R(stdout=f"{self._mtime}\n", exit_code=0)
        return _R(stdout="", exit_code=0)

    def _escape_shell_arg(self, arg: str) -> str:
        return "'" + arg.replace("'", "'\"'\"'") + "'"


@pytest.fixture(autouse=True)
def _reset_file_sessions() -> None:
    file_session._sessions.clear()
    yield
    file_session._sessions.clear()


@pytest.fixture
def stub_fileops() -> _StubFileOps:
    return _StubFileOps()


@pytest.fixture
def stub_session(stub_fileops: _StubFileOps) -> file_session.FileToolSession:
    sess = file_session.FileToolSession(file_ops=stub_fileops)  # type: ignore[arg-type]
    file_session._sessions["sess_01TEST"] = sess
    return sess


class TestArguments:
    async def test_missing_mode_raises(self, stub_session: file_session.FileToolSession) -> None:
        with pytest.raises(EditArgumentError):
            await edit_handler("sess_01TEST", {})

    async def test_invalid_mode_raises(self, stub_session: file_session.FileToolSession) -> None:
        with pytest.raises(EditArgumentError):
            await edit_handler("sess_01TEST", {"mode": "other"})

    async def test_replace_missing_path_raises(
        self, stub_session: file_session.FileToolSession
    ) -> None:
        with pytest.raises(EditArgumentError):
            await edit_handler(
                "sess_01TEST",
                {"mode": "replace", "old_string": "x", "new_string": "y"},
            )

    async def test_replace_missing_old_string_raises(
        self, stub_session: file_session.FileToolSession
    ) -> None:
        with pytest.raises(EditArgumentError):
            await edit_handler(
                "sess_01TEST",
                {"mode": "replace", "path": "/workspace/a.py", "new_string": "y"},
            )

    async def test_patch_missing_patch_raises(
        self, stub_session: file_session.FileToolSession
    ) -> None:
        with pytest.raises(EditArgumentError):
            await edit_handler("sess_01TEST", {"mode": "patch"})


class TestReplaceMode:
    async def test_happy_path(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        result = await edit_handler(
            "sess_01TEST",
            {
                "mode": "replace",
                "path": "/workspace/a.py",
                "old_string": "old",
                "new_string": "new",
            },
        )
        assert result["success"] is True
        assert "diff" in result
        assert stub_fileops.replace_calls == [("/workspace/a.py", "old", "new", False)]

    async def test_replace_all_flag_propagates(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        await edit_handler(
            "sess_01TEST",
            {
                "mode": "replace",
                "path": "/workspace/a.py",
                "old_string": "old",
                "new_string": "new",
                "replace_all": True,
            },
        )
        assert stub_fileops.replace_calls[0][3] is True

    async def test_sensitive_path_rejected(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        result = await edit_handler(
            "sess_01TEST",
            {
                "mode": "replace",
                "path": "/etc/passwd",
                "old_string": "x",
                "new_string": "y",
            },
        )
        assert "error" in result
        # No call was forwarded to the ops layer.
        assert stub_fileops.replace_calls == []

    async def test_no_match_error_propagates(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        stub_fileops.set_replace_result(
            PatchResult(success=False, error="Could not find match for old_string")
        )
        result = await edit_handler(
            "sess_01TEST",
            {
                "mode": "replace",
                "path": "/workspace/a.py",
                "old_string": "xyz",
                "new_string": "abc",
            },
        )
        assert result["success"] is False
        assert "error" in result


class TestPatchMode:
    async def test_happy_path(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /workspace/a.py\n"
            "@@ hunk @@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        result = await edit_handler("sess_01TEST", {"mode": "patch", "patch": patch})
        assert result["success"] is True
        assert stub_fileops.patch_v4a_calls == [patch]

    async def test_sensitive_target_rejects_whole_patch(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /workspace/a.py\n"
            "*** Update File: /etc/passwd\n"
            "*** End Patch\n"
        )
        result = await edit_handler("sess_01TEST", {"mode": "patch", "patch": patch})
        assert "error" in result
        assert "sensitive" in result["error"].lower()
        # Patch was never forwarded to the ops layer.
        assert stub_fileops.patch_v4a_calls == []

    async def test_add_file_target_checked(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        patch = "*** Begin Patch\n*** Add File: /etc/cron.d/evil\n+x\n*** End Patch\n"
        result = await edit_handler("sess_01TEST", {"mode": "patch", "patch": patch})
        assert "error" in result

    async def test_move_file_both_paths_checked(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        patch = "*** Begin Patch\n*** Move File: /workspace/a.py -> /etc/new.py\n*** End Patch\n"
        result = await edit_handler("sess_01TEST", {"mode": "patch", "patch": patch})
        assert "error" in result
