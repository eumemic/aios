"""Ownership-aware mkdir + worker-startup repair pass (#959).

The worker container runs as root; the api runs as uid 1000. Both write
under ``workspace_root``. A shared dir the worker creates first is
``root:root`` and the api (no CAP_CHOWN) can't write into it — the
``POST /v1/sessions/{id}/files`` bare-500. ``ensure_owned_dir`` chowns
newly-created components to the configured owner when running as root;
``repair_workspace_ownership`` fixes pre-existing root-owned residue on a
bounded frontier at worker startup.

No root needed: ``os.geteuid`` is monkeypatched and ``os.chown`` is
recorded rather than executed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aios.config import get_settings
from aios.sandbox.volumes import (
    ensure_owned_dir,
    ensure_session_uploads_dir,
)
from aios.sandbox.workspace_ownership import repair_workspace_ownership


@pytest.fixture
def _owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``workspace_root`` at a tmp dir and pin owner uid/gid to 1000."""
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    monkeypatch.setattr(settings, "workspaces_owner_uid", 1000)
    monkeypatch.setattr(settings, "workspaces_owner_gid", 1000)
    return tmp_path


def _record_chowns(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    chowns: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "chown", lambda p, u, g: chowns.append((str(p), u, g)))
    return chowns


def _record_lchowns(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    """Recorder for the REPAIR pass, which uses ``os.lchown`` (symlink-aware)
    so a root-owned symlink is chowned in place rather than following it to
    chown the target (see #959 hardening)."""
    lchowns: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "lchown", lambda p, u, g: lchowns.append((str(p), u, g)))
    return lchowns


class TestEnsureOwnedDir:
    def test_ensure_owned_dir_root_chowns_only_new_components(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chowns = _record_chowns(monkeypatch)
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        # workspace_root pre-exists (tmp_path is created by pytest).
        target = _owner / "_uploads" / "sess_X"

        result = ensure_owned_dir(target)

        assert result == target
        assert target.is_dir()
        chowned_paths = {p for p, _u, _g in chowns}
        # Exactly the two newly-created components — NOT workspace_root.
        assert chowned_paths == {str(_owner / "_uploads"), str(target)}

    def test_ensure_owned_dir_non_root_zero_chowns(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chowns = _record_chowns(monkeypatch)
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        target = _owner / "_uploads" / "sess_Y"

        result = ensure_owned_dir(target)

        assert result == target
        assert target.is_dir()
        assert chowns == []

    def test_ensure_owned_dir_second_call_no_op_chown(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chowns = _record_chowns(monkeypatch)
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        target = _owner / "_uploads" / "sess_Z"

        ensure_owned_dir(target)
        assert len(chowns) == 2  # _uploads + sess_Z
        chowns.clear()

        ensure_owned_dir(target)
        assert chowns == []  # nothing new created → nothing chowned

    def test_ensure_owned_dir_uses_configured_uid_gid(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspaces_owner_uid", 4242)
        monkeypatch.setattr(settings, "workspaces_owner_gid", 4343)
        chowns = _record_chowns(monkeypatch)
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        ensure_owned_dir(_owner / "_uploads" / "sess_W")

        assert chowns  # sanity: something was chowned
        assert all((u, g) == (4242, 4343) for _p, u, g in chowns)

    def test_ensure_session_uploads_dir_chowns_both_levels(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The traceback-site helper (volumes.py:335) routes through
        ``ensure_owned_dir``, so a fresh session chowns both ``_uploads``
        and ``_uploads/<session_id>`` under root."""
        chowns = _record_chowns(monkeypatch)
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        path = ensure_session_uploads_dir("sess_uploads")

        assert path.is_dir()
        chowned_paths = {p for p, _u, _g in chowns}
        assert str(_owner / "_uploads") in chowned_paths
        assert str(_owner / "_uploads" / "sess_uploads") in chowned_paths


class TestRepairWorkspaceOwnership:
    def test_repair_skips_when_not_root(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lchowns = _record_lchowns(monkeypatch)
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        # Build something repairable to prove it's the guard, not an empty tree.
        (_owner / "_uploads" / "sess_A").mkdir(parents=True)

        assert repair_workspace_ownership() == 0
        assert lchowns == []

    def test_repair_depth_bound_and_one_log_per_fix(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        # Make EVERY real on-disk dir read as mismatched so walk-scope is
        # the only variable.
        monkeypatch.setattr(settings, "workspaces_owner_uid", os.getuid() + 1)
        monkeypatch.setattr(settings, "workspaces_owner_gid", os.getgid() + 1)
        lchowns = _record_lchowns(monkeypatch)
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        (_owner / "_uploads" / "sess_A" / "file_deep").mkdir(parents=True)
        (_owner / "_memory_stores" / "store1" / "nested").mkdir(parents=True)
        (_owner / "acc_x" / "sess_b" / "deep").mkdir(parents=True)

        count = repair_workspace_ownership()

        chowned = {p for p, _u, _g in lchowns}
        expected = {
            str(_owner),
            str(_owner / "_uploads"),
            str(_owner / "_memory_stores"),
            str(_owner / "acc_x"),
            str(_owner / "_uploads" / "sess_A"),
            str(_owner / "_memory_stores" / "store1"),
            str(_owner / "acc_x" / "sess_b"),
        }
        assert chowned == expected
        # Depth bound: the third-level decoys are untouched.
        assert str(_owner / "_uploads" / "sess_A" / "file_deep") not in chowned
        assert str(_owner / "_memory_stores" / "store1" / "nested") not in chowned
        assert str(_owner / "acc_x" / "sess_b" / "deep") not in chowned
        assert count == len(expected)

    def test_repair_already_correct_owner_zero_chowns(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        # Every dir on disk already matches the configured owner.
        monkeypatch.setattr(settings, "workspaces_owner_uid", os.getuid())
        monkeypatch.setattr(settings, "workspaces_owner_gid", os.getgid())
        lchowns = _record_lchowns(monkeypatch)
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        (_owner / "_uploads" / "sess_A").mkdir(parents=True)
        (_owner / "acc_x" / "sess_b").mkdir(parents=True)

        assert repair_workspace_ownership() == 0
        assert lchowns == []

    def test_repair_continues_past_one_oserror(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspaces_owner_uid", os.getuid() + 1)
        monkeypatch.setattr(settings, "workspaces_owner_gid", os.getgid() + 1)
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        (_owner / "_uploads" / "sess_A").mkdir(parents=True)

        bad = str(_owner / "_uploads")
        lchowns: list[tuple[str, int, int]] = []

        def _lchown(p: object, u: int, g: int) -> None:
            if str(p) == bad:
                raise OSError("boom")
            lchowns.append((str(p), u, g))

        monkeypatch.setattr(os, "lchown", _lchown)

        # Must not raise; the bad entry is skipped but the rest process.
        count = repair_workspace_ownership()

        chowned = {p for p, _u, _g in lchowns}
        assert bad not in chowned
        assert str(_owner) in chowned
        assert str(_owner / "_uploads" / "sess_A") in chowned
        # The failed entry is not counted.
        assert count == len(chowned)

    @pytest.mark.skipif(
        os.geteuid() == 0,
        reason="under real root lstat/chown semantics differ; this exercises the "
        "non-root recorder path with geteuid mocked to 0",
    )
    def test_repair_lchowns_symlink_and_skips_descent_into_target(
        self, _owner: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hardening (#959, FIX #1 + #2): a symlink under a shared root is
        chowned in place via ``os.lchown`` (NOT ``os.chown``, which would
        follow it and chown the target), and the symlink's target directory's
        children are NOT enumerated/chowned (no descent into a symlinked dir).
        """
        settings = get_settings()
        # Mismatch every on-disk entry so the only variable is symlink handling.
        monkeypatch.setattr(settings, "workspaces_owner_uid", os.getuid() + 1)
        monkeypatch.setattr(settings, "workspaces_owner_gid", os.getgid() + 1)
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        # An external directory OUTSIDE workspace_root with a child that must
        # never be touched by the repair walk.
        external = _owner.parent / "external_target"
        (external / "secret_child").mkdir(parents=True)

        # A shared root containing a symlink that points at the external dir.
        uploads = _owner / "_uploads"
        uploads.mkdir(parents=True)
        link = uploads / "link"
        link.symlink_to(external, target_is_directory=True)

        lchowns: list[str] = []
        chowns: list[str] = []
        monkeypatch.setattr(os, "lchown", lambda p, u, g: lchowns.append(str(p)))
        monkeypatch.setattr(os, "chown", lambda p, u, g: chowns.append(str(p)))

        repair_workspace_ownership()

        # (a) The symlink itself was lchowned (in place), never chowned.
        assert str(link) in lchowns
        assert chowns == [], "os.chown must never be called — repair uses lchown"
        # (b) No descent into the symlink target: its external child is untouched.
        assert str(external / "secret_child") not in lchowns
        assert str(external) not in lchowns
