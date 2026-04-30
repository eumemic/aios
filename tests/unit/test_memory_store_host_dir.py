"""Tests for the per-store host-dir path conventions and lazy materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.sandbox.volumes import (
    memory_store_host_dir,
    memory_store_lock_path,
    memory_stores_root,
)


def test_root_under_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    root = memory_stores_root()
    assert root == (tmp_path / "_memory_stores").resolve()


def test_host_dir_keyed_by_store_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    a = memory_store_host_dir("memstore_AAA")
    b = memory_store_host_dir("memstore_BBB")
    assert a != b
    assert a.parent == b.parent == (tmp_path / "_memory_stores").resolve()


def test_lock_path_sibling_of_host_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    store_id = "memstore_XYZ"
    lock = memory_store_lock_path(store_id)
    host = memory_store_host_dir(store_id)
    assert lock.parent == host.parent
    assert lock.name == f"{store_id}.lock"


def test_host_dir_pure_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """memory_store_host_dir does not touch the filesystem."""
    from aios.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    p = memory_store_host_dir("memstore_NEW")
    assert not p.exists()
