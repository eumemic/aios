"""Unit coverage for :func:`aios.sandbox.volumes.read_plane_file` (jarbot#106).

The browser plane is a bind mount a (potentially compromised) container
writes, so every read the API/worker does from it must follow NO symlink at
ANY component — the TOCTOU the Phase-2 red-team flagged (F1): a resolve →
check → read sequence lets the container swap a checked component for a
symlink into another account's plane before the read. ``read_plane_file``
opens each component ``O_NOFOLLOW`` relative to its parent's dir fd, so a
symlink anywhere fails at open time and containment holds by construction.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aios.sandbox.volumes import read_plane_file


def _plane(tmp_path: Path) -> Path:
    plane = tmp_path / "acc_ME"
    (plane / "frames").mkdir(parents=True)
    return plane


def test_reads_a_regular_file(tmp_path: Path) -> None:
    plane = _plane(tmp_path)
    (plane / "frames" / "0.jpg").write_bytes(b"\xff\xd8data")
    assert read_plane_file(plane, "frames/0.jpg") == b"\xff\xd8data"


def test_missing_file_is_none(tmp_path: Path) -> None:
    assert read_plane_file(_plane(tmp_path), "frames/nope.jpg") is None


def test_absolute_ref_is_refused(tmp_path: Path) -> None:
    assert read_plane_file(_plane(tmp_path), "/etc/passwd") is None


def test_dotdot_ref_is_refused(tmp_path: Path) -> None:
    plane = _plane(tmp_path)
    (tmp_path / "secret").write_bytes(b"nope")
    assert read_plane_file(plane, "../secret") is None
    assert read_plane_file(plane, "frames/../../secret") is None


def test_empty_component_is_refused(tmp_path: Path) -> None:
    assert read_plane_file(_plane(tmp_path), "frames//0.jpg") is None


def test_nul_byte_ref_returns_none_never_raises(tmp_path: Path) -> None:
    """A NUL in the ref would make os.open raise ValueError (not OSError); the
    guard must catch it so a hostile ref can never raise out of the reader (the
    screenshot sink turns a raise into a caller-sandbox eviction)."""
    assert read_plane_file(_plane(tmp_path), "frames/0\x00.jpg") is None


def test_oversized_file_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file larger than the per-read cap is refused, not read whole — a
    compromised container cannot exhaust memory by planting one big file the
    5 Hz frame poll re-reads."""
    import aios.sandbox.volumes as volumes

    monkeypatch.setattr(volumes, "_PLANE_READ_MAX_BYTES", 1024)
    plane = _plane(tmp_path)
    (plane / "frames" / "0.jpg").write_bytes(b"x" * 4096)
    assert read_plane_file(plane, "frames/0.jpg") is None
    # A file at/under the cap still reads back.
    (plane / "frames" / "small.jpg").write_bytes(b"y" * 512)
    assert read_plane_file(plane, "frames/small.jpg") == b"y" * 512


def test_symlinked_leaf_is_refused(tmp_path: Path) -> None:
    """The leaf swap: the file itself is a symlink into another plane."""
    plane = _plane(tmp_path)
    victim = tmp_path / "acc_VICTIM" / "profile" / "Cookies"
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"cookie-jar")
    (plane / "frames" / "0.jpg").symlink_to(victim)
    assert read_plane_file(plane, "frames/0.jpg") is None


def test_symlinked_intermediate_dir_is_refused(tmp_path: Path) -> None:
    """The dir swap: an intermediate component is a symlink to another plane's
    dir. The old resolve-then-check on the leaf would have missed this once the
    escaped dir resolved self-consistently."""
    plane = tmp_path / "acc_ME"
    plane.mkdir()
    victim_frames = tmp_path / "acc_VICTIM" / "frames"
    victim_frames.mkdir(parents=True)
    (victim_frames / "0.jpg").write_bytes(b"victim")
    (plane / "frames").symlink_to(victim_frames)
    assert read_plane_file(plane, "frames/0.jpg") is None


def test_fifo_at_path_does_not_block(tmp_path: Path) -> None:
    """A FIFO planted at the path must not wedge the reader (O_NONBLOCK on the
    leaf open) and is rejected as a non-regular file."""
    plane = _plane(tmp_path)
    os.mkfifo(plane / "frames" / "0.jpg")
    assert read_plane_file(plane, "frames/0.jpg") is None


def test_directory_at_path_is_refused(tmp_path: Path) -> None:
    plane = _plane(tmp_path)
    (plane / "frames" / "sub").mkdir()
    assert read_plane_file(plane, "frames/sub") is None
