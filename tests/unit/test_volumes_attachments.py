"""Unit coverage for the attachments helpers in :mod:`aios.sandbox.volumes`."""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.config import get_settings
from aios.sandbox.volumes import (
    attachments_root,
    ensure_session_attachments_dir,
    resolve_to_host_path,
    session_attachments_dir,
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


class TestResolveToHostPath:
    """Mapping must be exact and only fire for known mount roots."""

    def test_workspace_root_maps_to_session_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        assert resolve_to_host_path("sess-1", "/workspace") == (tmp_path / "sess-1").resolve()

    def test_workspace_subpath(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspace_root", tmp_path)
        result = resolve_to_host_path("sess-1", "/workspace/sub/file.txt")
        assert result == (tmp_path / "sess-1").resolve() / "sub" / "file.txt"

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
