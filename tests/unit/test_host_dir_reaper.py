"""Unit tests for ``aios.harness.host_dir_reaper`` (#1192).

The reaper GCs two reserved host trees that otherwise grow monotonically
with session/run count (the 2026-06-16 floodgates disk-fill):

* ``_session_repos/<session_id>`` — RECONSTRUCTIBLE git working-tree clones.
* ``_runs/<run_id>``             — NON-reconstructible per-run scratch.

The load-bearing safety property (the bug PR #1193 shipped) is that the
keep-set is **DB liveness, NOT container presence**: a ``suspended`` run
loses its container (idle-released) while staying live in the DB, so a
container-presence keep-set would ``rmtree`` a LIVE run's ``/workspace``.
These tests assert the DB-liveness keep-set directly, never a container set:

* a ``suspended`` (non-terminal) run's ``_runs`` dir SURVIVES;
* a TERMINAL run's ``_runs`` dir is reaped;
* a DB-live session's ``_session_repos`` dir SURVIVES, a dead one is reaped;
* fail-closed: a DB error reaps nothing;
* the kill-switch disables the whole reaper;
* confinement: symlinks and too-fresh dirs are never touched, the root
  itself is never removed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness import host_dir_reaper
from aios.harness.host_dir_reaper import sweep_host_dirs


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point both reserved roots at tmpdirs and drop the age floor to 0.

    ``session_repos_root(id)`` / ``run_workspace_dir(id)`` are patched to
    return ``<root>/<id>`` so the reaper's ``.parent`` derivation lands on
    the tmp roots.
    """
    repos_root = tmp_path / "_session_repos"
    runs_root = tmp_path / "_runs"
    repos_root.mkdir()
    runs_root.mkdir()

    monkeypatch.setattr(host_dir_reaper, "session_repos_root", lambda sid: repos_root / sid)
    monkeypatch.setattr(host_dir_reaper, "run_workspace_dir", lambda rid: runs_root / rid)

    settings = MagicMock()
    settings.host_dir_reaper_enabled = True
    settings.host_dir_reaper_min_age_seconds = 0
    monkeypatch.setattr(host_dir_reaper, "get_settings", lambda: settings)

    return {"repos": repos_root, "runs": runs_root, "settings": settings}


def _mkdir_aged(parent: Path, name: str, *, age_s: float = 10_000.0) -> Path:
    """Create ``parent/name`` and back-date its mtime past any age floor."""
    d = parent / name
    d.mkdir()
    (d / "marker").write_text("scratch")
    old = time.time() - age_s
    import os

    os.utime(d, (old, old))
    return d


def _fake_pool() -> MagicMock:
    """Pool whose ``acquire()`` yields a context-managed MagicMock conn."""
    pool = MagicMock()

    class _Cm:
        async def __aenter__(self) -> Any:
            return MagicMock()

        async def __aexit__(self, *_a: Any) -> None:
            return None

    pool.acquire.return_value = _Cm()
    return pool


async def test_suspended_run_dir_survives_terminal_is_reaped(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-terminal (``suspended``) run keeps its scratch; a terminal one loses it.

    This is the inverse of the PR #1193 data-loss bug: the suspended run is
    container-ABSENT but DB-LIVE, and its NON-reconstructible ``_runs`` dir
    MUST be preserved.
    """
    suspended = _mkdir_aged(roots["runs"], "wfr_suspended")
    terminal = _mkdir_aged(roots["runs"], "wfr_done")

    # Only the terminal run id comes back from the terminal-status query.
    monkeypatch.setattr(
        host_dir_reaper.wf_queries,
        "unscoped_terminal_run_ids",
        AsyncMock(return_value={"wfr_done"}),
    )
    # No session repos present; the session liveness query is irrelevant here.
    monkeypatch.setattr(
        host_dir_reaper.queries,
        "unscoped_live_session_ids",
        AsyncMock(return_value=set()),
    )

    removed = await sweep_host_dirs(_fake_pool())

    assert removed == 1
    assert suspended.exists(), "a suspended (DB-live) run's _runs dir must survive"
    assert not terminal.exists(), "a terminal run's _runs dir must be reaped"


async def test_runs_absent_from_db_is_kept(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``_runs`` dir whose run row is absent (not positively terminal) is KEPT.

    Non-reconstructible scratch is only deleted on a POSITIVELY observed
    terminal status — never on the mere absence of confirmation.
    """
    orphan = _mkdir_aged(roots["runs"], "wfr_unknown")
    monkeypatch.setattr(
        host_dir_reaper.wf_queries,
        "unscoped_terminal_run_ids",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        host_dir_reaper.queries,
        "unscoped_live_session_ids",
        AsyncMock(return_value=set()),
    )

    removed = await sweep_host_dirs(_fake_pool())

    assert removed == 0
    assert orphan.exists()


async def test_session_repos_live_survives_dead_reaped(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconstructible ``_session_repos`` reaps on the positive DB-liveness keep-set."""
    live = _mkdir_aged(roots["repos"], "sess_live")
    dead = _mkdir_aged(roots["repos"], "sess_dead")

    monkeypatch.setattr(
        host_dir_reaper.queries,
        "unscoped_live_session_ids",
        AsyncMock(return_value={"sess_live"}),
    )
    monkeypatch.setattr(
        host_dir_reaper.wf_queries,
        "unscoped_terminal_run_ids",
        AsyncMock(return_value=set()),
    )

    removed = await sweep_host_dirs(_fake_pool())

    assert removed == 1
    assert live.exists(), "a DB-live session's clone dir must survive"
    assert not dead.exists(), "a dead/archived session's clone dir must be reaped"


async def test_kill_switch_disables_all_deletion(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``host_dir_reaper_enabled=False`` deletes nothing without a redeploy."""
    roots["settings"].host_dir_reaper_enabled = False
    dead_run = _mkdir_aged(roots["runs"], "wfr_done")
    dead_sess = _mkdir_aged(roots["repos"], "sess_dead")

    term = AsyncMock(return_value={"wfr_done"})
    live = AsyncMock(return_value=set())
    monkeypatch.setattr(host_dir_reaper.wf_queries, "unscoped_terminal_run_ids", term)
    monkeypatch.setattr(host_dir_reaper.queries, "unscoped_live_session_ids", live)

    removed = await sweep_host_dirs(_fake_pool())

    assert removed == 0
    assert dead_run.exists()
    assert dead_sess.exists()
    # Disabled means we never even query the DB.
    term.assert_not_awaited()
    live.assert_not_awaited()


async def test_db_error_fails_closed(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A liveness-query failure reaps NOTHING for that tree (fail-closed)."""
    import asyncpg

    dead_run = _mkdir_aged(roots["runs"], "wfr_done")
    dead_sess = _mkdir_aged(roots["repos"], "sess_dead")

    monkeypatch.setattr(
        host_dir_reaper.wf_queries,
        "unscoped_terminal_run_ids",
        AsyncMock(side_effect=asyncpg.PostgresError("boom")),
    )
    monkeypatch.setattr(
        host_dir_reaper.queries,
        "unscoped_live_session_ids",
        AsyncMock(side_effect=asyncpg.PostgresError("boom")),
    )

    removed = await sweep_host_dirs(_fake_pool())

    assert removed == 0
    assert dead_run.exists(), "fail-closed: never reap on a failed liveness read"
    assert dead_sess.exists()


async def test_fresh_dir_below_age_floor_is_kept(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dir younger than the age floor is skipped (provision/commit race guard)."""
    roots["settings"].host_dir_reaper_min_age_seconds = 3600
    fresh = roots["runs"] / "wfr_done"
    fresh.mkdir()  # mtime ~= now, well under the 1h floor

    monkeypatch.setattr(
        host_dir_reaper.wf_queries,
        "unscoped_terminal_run_ids",
        AsyncMock(return_value={"wfr_done"}),
    )
    monkeypatch.setattr(
        host_dir_reaper.queries,
        "unscoped_live_session_ids",
        AsyncMock(return_value=set()),
    )

    removed = await sweep_host_dirs(_fake_pool())

    assert removed == 0
    assert fresh.exists()


async def test_symlink_child_is_never_followed(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A symlink entry in a reserved root is skipped — its target may escape."""
    outside = tmp_path / "outside_target"
    outside.mkdir()
    (outside / "precious").write_text("do not touch")
    link = roots["runs"] / "wfr_done"
    link.symlink_to(outside)

    monkeypatch.setattr(
        host_dir_reaper.wf_queries,
        "unscoped_terminal_run_ids",
        AsyncMock(return_value={"wfr_done"}),
    )
    monkeypatch.setattr(
        host_dir_reaper.queries,
        "unscoped_live_session_ids",
        AsyncMock(return_value=set()),
    )

    removed = await sweep_host_dirs(_fake_pool())

    assert removed == 0
    assert outside.exists()
    assert (outside / "precious").exists()


async def test_root_itself_is_never_removed(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with everything reaped, the reserved roots survive for the next mount."""
    _mkdir_aged(roots["runs"], "wfr_done")
    _mkdir_aged(roots["repos"], "sess_dead")
    monkeypatch.setattr(
        host_dir_reaper.wf_queries,
        "unscoped_terminal_run_ids",
        AsyncMock(return_value={"wfr_done"}),
    )
    monkeypatch.setattr(
        host_dir_reaper.queries,
        "unscoped_live_session_ids",
        AsyncMock(return_value=set()),
    )

    await sweep_host_dirs(_fake_pool())

    assert roots["runs"].exists() and roots["runs"].is_dir()
    assert roots["repos"].exists() and roots["repos"].is_dir()


async def test_missing_root_is_a_noop(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No reserved root on disk ⇒ zero work, no crash."""
    import shutil

    shutil.rmtree(roots["runs"])
    shutil.rmtree(roots["repos"])
    monkeypatch.setattr(
        host_dir_reaper.wf_queries,
        "unscoped_terminal_run_ids",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        host_dir_reaper.queries,
        "unscoped_live_session_ids",
        AsyncMock(return_value=set()),
    )

    removed = await sweep_host_dirs(_fake_pool())

    assert removed == 0
