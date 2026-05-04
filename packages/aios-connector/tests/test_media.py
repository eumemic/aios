"""Unit coverage for ``resolve_sandbox_path``.

Mirrors the server-side ``resolve_to_host_path`` containment tests.
Both helpers must reject the same attacks; duplicating coverage
(rather than sharing an import) keeps the SDK independently
verifiable.
"""

from __future__ import annotations

from pathlib import Path

from aios_connector import resolve_sandbox_path


class TestResolveSandboxPath:
    def test_workspace_subpath(self, tmp_path: Path) -> None:
        result = resolve_sandbox_path(
            session_id="sess-1",
            sandbox_path="/workspace/foo.jpg",
            workspace_root=tmp_path,
        )
        assert result == (tmp_path / "sess-1" / "foo.jpg").resolve()

    def test_attachments_subpath(self, tmp_path: Path) -> None:
        result = resolve_sandbox_path(
            session_id="sess-1",
            sandbox_path="/mnt/attachments/echo/evt-photo.jpg",
            workspace_root=tmp_path,
        )
        assert result == (tmp_path / "_attachments" / "sess-1" / "echo" / "evt-photo.jpg").resolve()

    def test_workspace_root(self, tmp_path: Path) -> None:
        result = resolve_sandbox_path(
            session_id="sess-1",
            sandbox_path="/workspace",
            workspace_root=tmp_path,
        )
        assert result == (tmp_path / "sess-1").resolve()

    def test_unknown_paths_return_none(self, tmp_path: Path) -> None:
        for bad in (
            "/etc/passwd",
            "/tmp/x",
            "/mnt/memory/foo",
            "workspace/foo",
            "/workspaces/foo",
        ):
            assert (
                resolve_sandbox_path(
                    session_id="sess-1",
                    sandbox_path=bad,
                    workspace_root=tmp_path,
                )
                is None
            )

    def test_dotdot_escape_from_workspace_rejected(self, tmp_path: Path) -> None:
        assert (
            resolve_sandbox_path(
                session_id="sess-1",
                sandbox_path="/workspace/../../etc/hostname",
                workspace_root=tmp_path,
            )
            is None
        )

    def test_dotdot_escape_from_attachments_rejected(self, tmp_path: Path) -> None:
        assert (
            resolve_sandbox_path(
                session_id="sess-1",
                sandbox_path="/mnt/attachments/../foo",
                workspace_root=tmp_path,
            )
            is None
        )

    def test_innocuous_dotdot_inside_workspace_allowed(self, tmp_path: Path) -> None:
        result = resolve_sandbox_path(
            session_id="sess-1",
            sandbox_path="/workspace/sub/../foo.jpg",
            workspace_root=tmp_path,
        )
        assert result == (tmp_path / "sess-1" / "foo.jpg").resolve()

    def test_symlink_escape_via_workspace_rejected(self, tmp_path: Path) -> None:
        ws = (tmp_path / "sess-1").resolve()
        ws.mkdir(parents=True)
        outside = tmp_path / "outside.txt"
        outside.write_text("host-secret")
        (ws / "sneaky.jpg").symlink_to(outside)

        assert (
            resolve_sandbox_path(
                session_id="sess-1",
                sandbox_path="/workspace/sneaky.jpg",
                workspace_root=tmp_path,
            )
            is None
        )

    def test_symlink_target_inside_workspace_allowed(self, tmp_path: Path) -> None:
        ws = (tmp_path / "sess-1").resolve()
        ws.mkdir(parents=True)
        target = ws / "real.jpg"
        target.write_bytes(b"x")
        (ws / "alias.jpg").symlink_to(target)

        result = resolve_sandbox_path(
            session_id="sess-1",
            sandbox_path="/workspace/alias.jpg",
            workspace_root=tmp_path,
        )
        assert result == target.resolve()
