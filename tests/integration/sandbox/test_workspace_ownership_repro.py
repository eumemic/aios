"""Real-FS reproduction of #959 and end-to-end fix verification.

No DB / Docker. The bug: the worker (root) creates a shared dir under
``workspace_root`` as ``root:root``; the api (uid 1000) then can't mkdir
inside it → bare 500 on ``POST /v1/sessions/{id}/files``.

CI does not run as root, so we use ``chmod 0o555`` (read+exec, no write)
as a non-root proxy for "a dir the current process can't write into" —
the same ``PermissionError`` the api hits against a root-owned dir. The
``ensure_owned_dir`` chown fix and the ``repair_workspace_ownership``
pass are exercised with ``os.geteuid`` mocked to root and ``os.chown``
recorded (a non-root process can't really chown to another uid).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aios.config import get_settings
from aios.sandbox.volumes import (
    ensure_session_attachments_dir,
    ensure_session_uploads_dir,
)
from aios.sandbox.workspace_ownership import repair_workspace_ownership


@pytest.fixture
def _ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    monkeypatch.setattr(settings, "workspaces_owner_uid", 1000)
    monkeypatch.setattr(settings, "workspaces_owner_gid", 1000)
    return tmp_path


@pytest.mark.skipif(os.geteuid() == 0, reason="chmod barrier is bypassed by root")
def test_root_owned_uploads_blocks_api_mkdir_then_repair_fixes(_ws: Path) -> None:
    """chmod 0o555 stands in for the real root-owned-unwritable ``_uploads``:
    the api-side mkdir surfaces ``PermissionError`` (the bare-500
    condition); making it writable again lets it succeed."""
    uploads = _ws / "_uploads"
    uploads.mkdir()
    os.chmod(uploads, 0o555)  # read+exec, no write — the #959 failure mode
    try:
        with pytest.raises(PermissionError):
            ensure_session_uploads_dir("sess_z")
        # Repair-equivalent: restore writability (real fix would chown).
        os.chmod(uploads, 0o755)
        path = ensure_session_uploads_dir("sess_z")
        assert path.is_dir()
    finally:
        os.chmod(uploads, 0o755)  # ensure tmp cleanup can rmtree


def test_full_sequence_worker_create_repair_then_api_write(
    _ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker-create (root, chown recorded) → repair → api-side leaf
    mkdir succeeds against the writable tree."""
    # ── Phase 1: worker creates the shared tree as root; chown recorded.
    settings = get_settings()
    phase1_chowns: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "chown", lambda p, u, g: phase1_chowns.append((str(p), u, g)))
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    ensure_session_uploads_dir("sess_1")
    ensure_session_attachments_dir("sess_1")

    chowned = {p for p, _u, _g in phase1_chowns}
    assert str(_ws / "_uploads") in chowned
    assert str(_ws / "_uploads" / "sess_1") in chowned
    assert str(_ws / "_attachments") in chowned
    assert str(_ws / "_attachments" / "sess_1") in chowned
    assert all((u, g) == (1000, 1000) for _p, u, g in phase1_chowns)

    # ── Phase 2: repair pass over a tree whose on-disk owner mismatches.
    # The repair pass uses ``os.lchown`` (symlink-aware), so record that.
    monkeypatch.setattr(settings, "workspaces_owner_uid", os.getuid() + 1)
    monkeypatch.setattr(settings, "workspaces_owner_gid", os.getgid() + 1)
    phase2_chowns: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "lchown", lambda p, u, g: phase2_chowns.append((str(p), u, g)))

    repaired = repair_workspace_ownership()
    assert repaired > 0
    phase2_chowned = {p for p, _u, _g in phase2_chowns}
    # Bounded frontier: roots + their immediate children.
    assert str(_ws / "_uploads") in phase2_chowned
    assert str(_ws / "_uploads" / "sess_1") in phase2_chowned
    assert str(_ws / "_attachments") in phase2_chowned

    # ── Phase 3: api (uid 1000) writes a per-file leaf into the now-owned
    # tree. The real dirs on disk are owned by the test runner and writable,
    # so the bare leaf mkdir succeeds — no PermissionError.
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    leaf = ensure_session_uploads_dir("sess_1") / "file_id_abc"
    leaf.mkdir(exist_ok=False)
    assert leaf.is_dir()
