"""Idle host scratch-dir reaper (#1192).

Two reserved host trees grow monotonically with session/run count and have
no GC — manual ``rm`` was the only reclaim, which is what caused the
2026-06-16 floodgates disk-fill (``_session_repos`` reached 62GB across 150
dirs → ``/`` to 100% → postgres ``PANIC: No space left on device`` → whole
aios runtime down):

* ``<workspace_root>/_session_repos/<session_id>/`` — per-session git
  working-tree clones of ``github_repository`` attachments.
* ``<workspace_root>/_runs/<run_id>/`` — per-run workflow scratch
  (``/workspace`` for a run sandbox).

This reaper closes that missing-GC gap. It runs an immediate sweep at worker
startup and then periodically (mirroring the snapshot GC reconciler).

Safety model — DB liveness, NOT container presence
--------------------------------------------------
The keep-set is derived from **DB liveness**, never from running-container
presence. A session whose container the idle reaper released
(``container_idle_timeout_seconds``), or a run that is gate-``suspended``,
loses its container while staying live in the DB. Keying the keep-set on
``docker ps`` would reap a LIVE owner's dir — the data-loss bug PR #1193
shipped (a seat review caught it deleting a suspended run's ``/workspace``).

The two trees are treated **asymmetrically** by *reconstructibility*:

* ``_session_repos`` is RECONSTRUCTIBLE — github-clone rmtree+re-clones the
  working tree on the next provision unconditionally. So it reaps on the
  positive keep-set: keep iff the session is DB-live (row exists, not
  archived); reap otherwise. A wrongly-reaped clone is just re-cloned.

* ``_runs`` is NOT reconstructible — the per-run ``/workspace`` scratch is
  ephemeral with no re-derivation path. So it reaps ONLY on a *positively
  observed* TERMINAL run status (``completed``/``errored``/``cancelled``). A
  ``suspended`` (or ``pending``/``running``, or absent) run is KEPT. We never
  delete non-reconstructible scratch on the mere absence of confirmation.

Per-tree liveness is re-derived from a fresh DB read **at sweep time**, after
the on-disk enumeration and after the age floor — not frozen once globally —
so the docstrings here describe the property the code actually has.

Confinement (kept from PR #1193, these were sound)
--------------------------------------------------
* Operate only on direct ``iterdir()`` children of each resolved reserved
  root; never string-join an id onto a path.
* Each candidate must resolve to a real child *of that resolved root* and
  must not be a symlink — a symlink whose target escapes the root is skipped.
* The roots themselves are never deleted; an empty root is left in place so
  the bind-mount source survives for the next provision.

Kill-switch / fail-closed
-------------------------
``host_dir_reaper_enabled`` (default-on) disables the whole reaper without a
redeploy — irreversible host deletion on a worker that just had a P1
disk-fill must be disableable. On ANY DB error deriving liveness, the sweep
deletes nothing this pass (fail-closed): a DB hiccup must never be read as
"everything is dead".
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.db.queries import workflows as wf_queries
from aios.logging import get_logger
from aios.sandbox.volumes import run_workspace_dir, session_repos_root

log = get_logger("aios.harness.host_dir_reaper")


@dataclass(frozen=True, slots=True)
class _Candidate:
    """An on-disk reserved-tree child the sweep is considering for reaping.

    ``owner_id`` is the session id (``_session_repos``) or run id
    (``_runs``) parsed off the directory name — used only to look the owner
    up in the DB liveness set; it is never re-joined onto a path.
    """

    owner_id: str
    path: Path


def _scan_children(root: Path, *, min_age_seconds: float, now: float) -> list[_Candidate]:
    """Enumerate reap-eligible direct children of ``root`` (confinement gate).

    A child is a candidate iff it (a) resolves to a real directory that is a
    direct child of the *resolved* ``root``, (b) is not a symlink, and (c) is
    older than ``min_age_seconds`` by mtime (the provision/commit race floor).
    The root itself is never returned. A missing root yields no candidates.
    """
    resolved_root = root.resolve()
    if not resolved_root.exists():
        return []
    cutoff = now - min_age_seconds
    candidates: list[_Candidate] = []
    for child in resolved_root.iterdir():
        # Never follow a symlink: its target may escape the reserved root.
        if child.is_symlink():
            continue
        if not child.is_dir():
            continue
        # Confinement: the child must resolve to a real child of the resolved
        # root (defence in depth on top of the symlink skip above).
        resolved_child = child.resolve()
        if resolved_child.parent != resolved_root:
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if mtime > cutoff:
            continue  # too fresh — may be an in-flight provision
        candidates.append(_Candidate(owner_id=child.name, path=child))
    return candidates


def _reap(candidates: list[_Candidate], reap_ids: set[str]) -> int:
    """rmtree every candidate whose ``owner_id`` is in ``reap_ids``.

    Returns the count actually removed. Per-tree errors are logged and
    swallowed — one un-removable dir (perm drift, FS read-only) must not
    abort the rest of the sweep, and the next sweep retries it.
    """
    removed = 0
    for cand in candidates:
        if cand.owner_id not in reap_ids:
            continue
        try:
            shutil.rmtree(cand.path)
            removed += 1
        except OSError:
            log.exception("host_dir_reaper.rmtree_failed", path=str(cand.path))
    return removed


async def _reap_session_repos(
    pool: asyncpg.Pool[Any], *, min_age_seconds: float, now: float
) -> int:
    """Reap idle ``_session_repos/<session_id>`` clones.

    Reconstructible ⇒ reap on the positive keep-set: a candidate is reaped
    iff its session is NOT DB-live (deleted or archived). Liveness is read
    fresh here (not a frozen global snapshot). Fail-closed: a DB error reaps
    nothing this pass.
    """
    # session_repos_root takes a session id only to build the per-session
    # path; .parent is the reserved ``_session_repos`` root we scan.
    root = session_repos_root("_").parent
    candidates = _scan_children(root, min_age_seconds=min_age_seconds, now=now)
    if not candidates:
        return 0
    owner_ids = [c.owner_id for c in candidates]
    try:
        async with pool.acquire() as conn:
            live = await queries.unscoped_live_session_ids(conn, owner_ids)
    except (asyncpg.PostgresError, OSError):
        log.exception("host_dir_reaper.session_repos_liveness_failed")
        return 0  # fail-closed: never reap on a failed liveness read
    reap_ids = {oid for oid in owner_ids if oid not in live}
    removed = _reap(candidates, reap_ids)
    log.info(
        "host_dir_reaper.session_repos_swept",
        candidates=len(candidates),
        removed=removed,
        kept=len(candidates) - removed,
    )
    return removed


async def _reap_runs(pool: asyncpg.Pool[Any], *, min_age_seconds: float, now: float) -> int:
    """Reap ``_runs/<run_id>`` scratch for TERMINAL runs ONLY.

    NOT reconstructible ⇒ reap ONLY on a positively observed terminal status;
    a suspended/running/pending/absent run is kept. Fail-closed on DB error.
    """
    root = run_workspace_dir("_").parent
    candidates = _scan_children(root, min_age_seconds=min_age_seconds, now=now)
    if not candidates:
        return 0
    owner_ids = [c.owner_id for c in candidates]
    try:
        async with pool.acquire() as conn:
            terminal = await wf_queries.unscoped_terminal_run_ids(conn, owner_ids)
    except (asyncpg.PostgresError, OSError):
        log.exception("host_dir_reaper.runs_liveness_failed")
        return 0  # fail-closed
    removed = _reap(candidates, terminal)
    log.info(
        "host_dir_reaper.runs_swept",
        candidates=len(candidates),
        removed=removed,
        kept=len(candidates) - removed,
    )
    return removed


async def sweep_host_dirs(pool: asyncpg.Pool[Any]) -> int:
    """One reaper sweep over ``_session_repos`` and ``_runs``.

    Returns the total directories removed. Honors the kill-switch
    (``host_dir_reaper_enabled``): when disabled, deletes nothing and returns
    0. Each tree is independent — a failure deriving one tree's liveness
    fail-closes only that tree.
    """
    settings = get_settings()
    if not settings.host_dir_reaper_enabled:
        return 0
    min_age = float(settings.host_dir_reaper_min_age_seconds)
    now = time.time()
    repos = await _reap_session_repos(pool, min_age_seconds=min_age, now=now)
    runs = await _reap_runs(pool, min_age_seconds=min_age, now=now)
    return repos + runs
