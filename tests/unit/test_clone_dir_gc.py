"""Unit tests for ``aios.harness.clone_dir_gc`` (#1192).

The reaper must:

- Remove idle ``_session_repos/<sess>`` and ``_runs/<wfr>`` dirs (no live
  container, mtime older than the threshold).
- NEVER reclaim a dir whose owner has a running sandbox container — the
  keep-set is derived from ``backend.list_managed`` at delete time.
- Spare dirs younger than the age floor (a session between containers but
  about to wake).
- Fail closed if the backend listing raises (cannot compute keep-set → do
  not delete anything).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aios.harness.clone_dir_gc import reap_idle_clone_dirs
from aios.sandbox.backends.base import ManagedSandboxRef


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the volume helpers + settings at a tmp workspace root."""
    root = tmp_path / "workspaces"
    (root / "_session_repos").mkdir(parents=True)
    (root / "_runs").mkdir(parents=True)

    monkeypatch.setattr(
        "aios.harness.clone_dir_gc.session_repos_gc_root",
        lambda: root / "_session_repos",
    )
    monkeypatch.setattr(
        "aios.harness.clone_dir_gc.runs_root",
        lambda: root / "_runs",
    )

    class _FakeSettings:
        clone_dir_gc_idle_age_seconds = 3600
        instance_id = "test-instance"

    monkeypatch.setattr(
        "aios.harness.clone_dir_gc.get_settings",
        lambda: _FakeSettings(),
    )
    return root


def _seed_dir(root: Path, tree: str, owner_id: str, *, age_seconds: float) -> Path:
    """Create ``<root>/<tree>/<owner_id>`` with a backdated mtime."""
    target = root / tree / owner_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "ghrepo_x").mkdir(exist_ok=True)
    (target / "ghrepo_x" / "file").write_text("data")
    past = time.time() - age_seconds
    os.utime(target, (past, past))
    return target


def _backend(refs: list[ManagedSandboxRef]) -> AsyncMock:
    backend = AsyncMock()
    backend.list_managed = AsyncMock(return_value=refs)
    return backend


async def test_removes_idle_session_repo_and_run_dirs(workspace_root: Path) -> None:
    sess = _seed_dir(workspace_root, "_session_repos", "sess_idle", age_seconds=7200)
    run = _seed_dir(workspace_root, "_runs", "wfr_idle", age_seconds=7200)
    backend = _backend([])  # no live containers

    result = await reap_idle_clone_dirs(backend)

    assert not sess.exists()
    assert not run.exists()
    assert result.session_repos_removed == 1
    assert result.runs_removed == 1
    assert result.failures == 0


async def test_never_reclaims_session_with_running_container(workspace_root: Path) -> None:
    live = _seed_dir(workspace_root, "_session_repos", "sess_live", age_seconds=99999)
    run_live = _seed_dir(workspace_root, "_runs", "wfr_live", age_seconds=99999)
    backend = _backend(
        [
            ManagedSandboxRef(sandbox_id="c1", session_id="sess_live", running=True),
            ManagedSandboxRef(sandbox_id="c2", session_id="wfr_live", running=True),
        ]
    )

    result = await reap_idle_clone_dirs(backend)

    assert live.exists()
    assert run_live.exists()
    assert result.total_removed == 0


async def test_stopped_corpse_does_not_pin_clone_dir(workspace_root: Path) -> None:
    corpse = _seed_dir(workspace_root, "_session_repos", "sess_corpse", age_seconds=7200)
    backend = _backend(
        [ManagedSandboxRef(sandbox_id="c1", session_id="sess_corpse", running=False)]
    )

    result = await reap_idle_clone_dirs(backend)

    assert not corpse.exists()
    assert result.session_repos_removed == 1


async def test_spares_dirs_younger_than_age_floor(workspace_root: Path) -> None:
    fresh = _seed_dir(workspace_root, "_session_repos", "sess_fresh", age_seconds=60)
    backend = _backend([])

    result = await reap_idle_clone_dirs(backend)

    assert fresh.exists()
    assert result.total_removed == 0


async def test_fails_closed_when_listing_raises(workspace_root: Path) -> None:
    sess = _seed_dir(workspace_root, "_session_repos", "sess_idle", age_seconds=7200)
    backend = AsyncMock()
    backend.list_managed = AsyncMock(side_effect=RuntimeError("docker ps failed"))

    with pytest.raises(RuntimeError):
        await reap_idle_clone_dirs(backend)

    # Nothing deleted: keep-set could not be computed.
    assert sess.exists()


async def test_ignores_non_dir_entries(workspace_root: Path) -> None:
    (workspace_root / "_session_repos" / "stray.lock").write_text("x")
    backend = _backend([])

    result = await reap_idle_clone_dirs(backend)

    assert (workspace_root / "_session_repos" / "stray.lock").exists()
    assert result.total_removed == 0
