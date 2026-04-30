"""Tests for the atomic_mirror primitives used by memory-store sync."""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.sandbox.atomic_mirror import atomic_delete, atomic_write


def test_atomic_write_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.md"
    atomic_write(target, "hello")
    assert target.read_text() == "hello"


def test_atomic_write_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "x.md"
    target.write_text("v1")
    atomic_write(target, "v2")
    assert target.read_text() == "v2"


def test_atomic_write_no_temp_leftover(tmp_path: Path) -> None:
    target = tmp_path / "x.md"
    atomic_write(target, "ok")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp.")]
    assert leftovers == []


def test_atomic_delete_missing_is_ok(tmp_path: Path) -> None:
    atomic_delete(tmp_path / "never-existed.md")  # no exception


def test_atomic_delete_removes(tmp_path: Path) -> None:
    target = tmp_path / "x.md"
    target.write_text("bye")
    atomic_delete(target)
    assert not target.exists()


def test_atomic_write_handles_unicode(tmp_path: Path) -> None:
    target = tmp_path / "u.md"
    atomic_write(target, "héllo · 你好")
    assert target.read_text() == "héllo · 你好"


def test_atomic_write_cleans_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If os.replace fails, the temp file should be cleaned up."""
    import os

    real_replace = os.replace

    def boom(_src: str | Path, _dst: str | Path) -> None:
        raise OSError("simulated failure")

    target = tmp_path / "x.md"
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write(target, "content")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp.")]
    assert leftovers == []
    monkeypatch.setattr(os, "replace", real_replace)
