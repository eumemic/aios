"""Unit coverage for the attachments helpers in :mod:`aios.sandbox.volumes`."""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.config import get_settings
from aios.sandbox.volumes import (
    attachments_root,
    ensure_session_attachments_dir,
    ensure_session_uploads_dir,
    resolve_to_host_path,
    session_attachments_dir,
    session_uploads_dir,
    uploads_root,
)


def test_attachments_root_under_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    assert attachments_root() == (tmp_path / "_attachments").resolve()


def test_session_attachments_dir_keyed_by_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    a = session_attachments_dir("sess-a")
    b = session_attachments_dir("sess-b")
    assert a != b
    assert a.parent == b.parent == (tmp_path / "_attachments").resolve()


def test_session_attachments_dir_pure_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    p = session_attachments_dir("sess-new")
    assert not p.exists()


def test_ensure_session_attachments_dir_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    p = ensure_session_attachments_dir("sess-1")
    assert p.exists() and p.is_dir()
    # Idempotent.
    p2 = ensure_session_attachments_dir("sess-1")
    assert p == p2


def test_uploads_root_under_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    assert uploads_root() == (tmp_path / "_uploads").resolve()


def test_session_uploads_dir_keyed_by_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    a = session_uploads_dir("sess-a")
    b = session_uploads_dir("sess-b")
    assert a != b
    assert a.parent == b.parent == (tmp_path / "_uploads").resolve()


def test_session_uploads_dir_pure_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    p = session_uploads_dir("sess-new")
    assert not p.exists()


def test_ensure_session_uploads_dir_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    p = ensure_session_uploads_dir("sess-1")
    assert p.exists() and p.is_dir()
    # Idempotent.
    p2 = ensure_session_uploads_dir("sess-1")
    assert p == p2


class TestResolveToHostPath:
    """Mapping must be exact and only fire for known mount roots."""

    def test_workspace_root_maps_to_session_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        # Legacy pre-#409 layout: caller supplies the legacy workspace_path.
        legacy = (tmp_path / "sess-1").resolve()
        assert resolve_to_host_path("sess-1", "/workspace", workspace_path=legacy) == legacy

    def test_workspace_subpath(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        legacy = (tmp_path / "sess-1").resolve()
        result = resolve_to_host_path("sess-1", "/workspace/sub/file.txt", workspace_path=legacy)
        assert result == legacy / "sub" / "file.txt"

    def test_attachments_root_maps_to_session_attachments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        result = resolve_to_host_path("sess-1", "/mnt/attachments")
        assert result == (tmp_path / "_attachments" / "sess-1").resolve()

    def test_attachments_subpath(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        result = resolve_to_host_path("sess-1", "/mnt/attachments/echo/evt-photo.jpg")
        assert result == (
            (tmp_path / "_attachments" / "sess-1").resolve() / "echo" / "evt-photo.jpg"
        )

    def test_uploads_root_maps_to_session_uploads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        result = resolve_to_host_path("sess-1", "/mnt/uploads")
        assert result == (tmp_path / "_uploads" / "sess-1").resolve()

    def test_uploads_subpath(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        result = resolve_to_host_path("sess-1", "/mnt/uploads/file_01XYZ/photo.png")
        assert result == ((tmp_path / "_uploads" / "sess-1").resolve() / "file_01XYZ" / "photo.png")

    def test_uploads_dotdot_escape_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        # Same containment guarantee as workspace/attachments — symmetric protection.
        assert resolve_to_host_path("sess-1", "/mnt/uploads/../foo") is None

    def test_unknown_paths_return_none(self) -> None:
        # Paths outside the bind-mount roots fall back to docker-exec.
        assert resolve_to_host_path("sess-1", "/etc/hostname") is None
        assert resolve_to_host_path("sess-1", "/tmp/x") is None
        # Memory mounts handled separately — not in scope for this helper.
        assert resolve_to_host_path("sess-1", "/mnt/memory/foo/bar") is None
        # No prefix match (relative path).
        assert resolve_to_host_path("sess-1", "workspace/foo") is None
        # Not /workspace itself but starts with the literal prefix.
        assert resolve_to_host_path("sess-1", "/workspaces/foo") is None

    def test_post_409_workspace_uses_explicit_workspace_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Post-#409 layout: nested ``<workspace_root>/<account>/<session>``
        passed in as ``workspace_path`` and used as the base."""
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        nested = (tmp_path / "acct-1" / "sess-1").resolve()
        nested.mkdir(parents=True)
        result = resolve_to_host_path("sess-1", "/workspace/foo.png", workspace_path=nested)
        assert result == nested / "foo.png"

    def test_pre_409_workspace_layout_still_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-#409 layout: caller supplies the legacy
        ``<workspace_root>/<session>`` shape and it resolves fine."""
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        legacy = (tmp_path / "sess-1").resolve()
        result = resolve_to_host_path("sess-1", "/workspace/file.txt", workspace_path=legacy)
        assert result == legacy / "file.txt"

    def test_workspace_path_missing_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail-closed: ``/workspace*`` without ``workspace_path`` returns None.

        No silent fallback to the legacy synthetic path.
        """
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        assert resolve_to_host_path("sess-1", "/workspace/foo") is None
        assert resolve_to_host_path("sess-1", "/workspace") is None

    def test_attachments_branch_unaffected_by_workspace_path_param(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        without = resolve_to_host_path("sess-1", "/mnt/attachments/echo/x.jpg")
        with_ws = resolve_to_host_path(
            "sess-1",
            "/mnt/attachments/echo/x.jpg",
            workspace_path=tmp_path / "acct-1" / "sess-1",
        )
        assert without == with_ws

    def test_uploads_branch_unaffected_by_workspace_path_param(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        without = resolve_to_host_path("sess-1", "/mnt/uploads/f/x.png")
        with_ws = resolve_to_host_path(
            "sess-1",
            "/mnt/uploads/f/x.png",
            workspace_path=tmp_path / "acct-1" / "sess-1",
        )
        assert without == with_ws

    def test_workspace_path_with_workspace_exact_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        nested = (tmp_path / "acct-1" / "sess-1").resolve()
        nested.mkdir(parents=True)
        result = resolve_to_host_path("sess-1", "/workspace", workspace_path=nested)
        assert result == nested


class TestResolveToHostPathTraversal:
    """Containment check: model-controlled paths must not escape the
    bind-mount root via ``..`` normalization or symlink dereferencing."""

    def test_dotdot_escape_from_workspace_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        # `/workspace/../../etc/hostname` resolves outside the session's
        # workspace dir → must be rejected.
        legacy = (tmp_path / "sess-1").resolve()
        assert (
            resolve_to_host_path("sess-1", "/workspace/../../etc/hostname", workspace_path=legacy)
            is None
        )

    def test_dotdot_escape_from_attachments_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        assert resolve_to_host_path("sess-1", "/mnt/attachments/../foo") is None

    def test_deeply_nested_dotdot_escape_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        legacy = (tmp_path / "sess-1").resolve()
        assert (
            resolve_to_host_path(
                "sess-1",
                "/workspace/sub/../../../../etc/passwd",
                workspace_path=legacy,
            )
            is None
        )

    def test_innocuous_dotdot_inside_workspace_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``..`` that stays inside the workspace after normalization is fine."""
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        legacy = (tmp_path / "sess-1").resolve()
        result = resolve_to_host_path("sess-1", "/workspace/sub/../foo.jpg", workspace_path=legacy)
        assert result == legacy / "foo.jpg"

    def test_dotdot_containment_with_explicit_workspace_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``..`` escape rejected against the explicit (post-#409) base."""
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        nested = (tmp_path / "acct-1" / "sess-1").resolve()
        nested.mkdir(parents=True)
        assert (
            resolve_to_host_path("sess-1", "/workspace/../../etc/passwd", workspace_path=nested)
            is None
        )

    def test_symlink_escape_via_workspace_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A symlink inside ``/workspace`` whose target lives outside the
        bind mount must be rejected.  This covers the real attack: the
        model creates the symlink via ``bash`` inside the sandbox, then
        ``read``s through it to bypass containment."""
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        # Create the workspace dir (the bind-mount source) and an
        # outside-the-workspace target.
        ws = (tmp_path / "sess-1").resolve()
        ws.mkdir(parents=True)
        outside = tmp_path / "outside.txt"
        outside.write_text("host-secret")
        (ws / "sneaky.jpg").symlink_to(outside)

        assert resolve_to_host_path("sess-1", "/workspace/sneaky.jpg", workspace_path=ws) is None

    def test_symlink_target_inside_workspace_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A symlink whose target stays inside the workspace is fine."""
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        ws = (tmp_path / "sess-1").resolve()
        ws.mkdir(parents=True)
        target = ws / "real.jpg"
        target.write_bytes(b"x")
        (ws / "alias.jpg").symlink_to(target)

        result = resolve_to_host_path("sess-1", "/workspace/alias.jpg", workspace_path=ws)
        assert result == target.resolve()
