"""Idle host-clone-dir reaper for ``_session_repos`` and ``_runs`` (#1192).

The aios worker reaps orphaned sandbox *containers* at startup, but nothing
collected the host-side per-session / per-run working directories that back
them:

- ``<workspace_root>/_session_repos/<session_id>/`` — per-session git
  working-tree clones of ``github_repository`` resources. The github-clone
  provisioner (:mod:`aios.sandbox.github_clone`) ``rmtree``s and re-clones the
  working tree on **every** provision, so an idle session's clone is pure
  reconstructible cache: deleting it is exactly what aios does itself on the
  next wake.
- ``<workspace_root>/_runs/<run_id>/`` — per-run dev_pipeline scratch dirs.
  A run sandbox is ephemeral scratch with no durable rootfs.

Without GC these grow monotonically with session/run count and are reclaimed
only by manual intervention. On 2026-06-16 ``_session_repos`` reached 62GB
across 150 dirs and filled ``/`` to 100%, crashing postgres (``PANIC: No space
left on device``) and the whole aios runtime.

This reaper mirrors the orphan-container reaper: it runs at worker startup and
on a periodic sweep. The keep-set — owners (sessions/runs) with a LIVE sandbox
container — is re-derived from the live sandbox registry (``docker ps`` via
:meth:`SandboxBackend.list_managed`) **at delete time**, so a session that woke
mid-sweep and re-provisioned a container is spared (race guard). A dir whose
owner has a running container is NEVER reclaimed, regardless of age. Only dirs
with no live container AND older than ``clone_dir_gc_idle_age_seconds`` (by
mtime) are removed.

``SandboxSpec.session_id`` is the opaque owner-label field carrying a
session-OR-run id (``sess_…`` or ``wfr_…``); the backend stamps it on every
managed container, so a single keep-set covers both directory trees — the
``_session_repos`` leaf is the session id and the ``_runs`` leaf is the run id,
and both are owner ids.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from aios.config import get_settings
from aios.logging import get_logger
from aios.sandbox.backends.base import SandboxBackend
from aios.sandbox.volumes import runs_root, session_repos_gc_root

log = get_logger("aios.harness.clone_dir_gc")


@dataclass(frozen=True, slots=True)
class CloneDirGcResult:
    """Outcome of one reap pass: counts of removed dirs per tree + failures."""

    session_repos_removed: int = 0
    runs_removed: int = 0
    failures: int = 0

    @property
    def total_removed(self) -> int:
        return self.session_repos_removed + self.runs_removed


async def _live_owner_ids(backend: SandboxBackend, *, instance_id: str) -> set[str]:
    """Owner ids (session/run) with a LIVE container for this worker instance.

    Re-derived from ``backend.list_managed`` at every call so it can be
    re-checked AT DELETE TIME against the same listing that drove the scan.
    Only ``running`` containers count: a stopped corpse does not pin its
    clone dir (the salvage/GC path will remove the corpse, and the dir is
    reconstructible). A ref whose ``session_id`` (owner label) is ``None`` —
    a container missing the label — cannot be attributed to any owner dir, so
    it is skipped; it pins nothing. That is safe: an unlabelled container is
    not supposed to happen, and the age floor still applies to every dir.
    """
    refs = await backend.list_managed(instance_id=instance_id)
    return {ref.session_id for ref in refs if ref.running and ref.session_id is not None}


def _reap_tree(
    root: Path,
    *,
    live_owner_ids: set[str],
    idle_age_seconds: float,
    now: float,
    tree_label: str,
) -> tuple[int, int]:
    """Remove idle child dirs of ``root``; return ``(removed, failures)``.

    A child dir is removed iff its leaf name (the owner id) is NOT in
    ``live_owner_ids`` AND its mtime is older than ``idle_age_seconds``. The
    parent ``root`` is never removed (it is the bind-mount-source parent and
    re-created lazily anyway). ``rmtree`` failures are counted and logged —
    a perm-drift / read-only-FS failure is a real signal, not silently
    swallowed — but never abort the sweep of sibling dirs.
    """
    if not root.exists():
        return 0, 0
    removed = 0
    failures = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        owner_id = child.name
        if owner_id in live_owner_ids:
            continue
        try:
            age = now - child.stat().st_mtime
        except OSError as err:
            failures += 1
            log.warning(
                "clone_dir_gc.stat_failed", tree=tree_label, path=str(child), error=str(err)
            )
            continue
        if age < idle_age_seconds:
            continue
        try:
            shutil.rmtree(child)
            removed += 1
            log.info(
                "clone_dir_gc.removed",
                tree=tree_label,
                owner_id=owner_id,
                age_seconds=round(age),
            )
        except OSError as err:
            failures += 1
            log.warning(
                "clone_dir_gc.rmtree_failed",
                tree=tree_label,
                path=str(child),
                error=str(err),
            )
    return removed, failures


async def reap_idle_clone_dirs(backend: SandboxBackend) -> CloneDirGcResult:
    """One reap pass over ``_session_repos`` and ``_runs`` host dirs.

    Removes per-session clone dirs and per-run scratch dirs whose owner has
    NO live sandbox container (this worker instance) and whose mtime is older
    than ``clone_dir_gc_idle_age_seconds``. The live-owner keep-set is derived
    ONCE at the top of the pass from ``backend.list_managed`` — a single
    ``docker ps`` — and used for both trees, which is the race guard the issue
    requires: a session that re-provisioned a container after the scan started
    is in the keep-set and is spared.

    A session/run with a running sandbox is never reclaimed. Returns the
    per-tree removal counts; the caller logs them. Backend listing failures
    propagate (a failed ``docker ps`` means we cannot safely compute the
    keep-set, so we must NOT delete anything — fail closed).
    """
    settings = get_settings()
    idle_age = float(settings.clone_dir_gc_idle_age_seconds)
    instance_id = settings.instance_id

    session_root = session_repos_gc_root()
    run_root = runs_root()
    if not session_root.exists() and not run_root.exists():
        return CloneDirGcResult()

    live = await _live_owner_ids(backend, instance_id=instance_id)
    now = time.time()

    session_removed, session_failures = _reap_tree(
        session_root,
        live_owner_ids=live,
        idle_age_seconds=idle_age,
        now=now,
        tree_label="_session_repos",
    )
    runs_removed, runs_failures = _reap_tree(
        run_root,
        live_owner_ids=live,
        idle_age_seconds=idle_age,
        now=now,
        tree_label="_runs",
    )
    return CloneDirGcResult(
        session_repos_removed=session_removed,
        runs_removed=runs_removed,
        failures=session_failures + runs_failures,
    )
