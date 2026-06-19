"""Archived-session ``/workspace`` reaper (#40 — the "45G hole").

The per-session workspace dir (``<workspace_root>/<account_id>/<session_id>``,
bind-mounted into the container at ``/workspace``) is **never deleted** when a
session ends — ``volumes.py`` calls cleanup "a Phase 6 polish item" that was
never built. Across thousands of ended sessions this reached ~45GB on Server B
and is the largest un-reaped store the #1192 reaper does NOT cover (that one
handles ``_session_repos`` clones and ``_runs`` scratch, not the workspace).

This reaper closes that gap. It is the P1 piece of the disk control loop
(``eumemic-company/architecture/disk-control-loop.md``).

What is reaped vs. what is sacred (never-delete — sessions ARE the company)
---------------------------------------------------------------------------
REAPED: the **workspace files on disk** — the ephemeral repo checkout + scratch
the session wrote under ``/workspace`` while running. For an *archived* session
these are terminal: nothing ever re-reads them.

NOT REAPED — sacred durable state, none of which lives under the workspace dir:
the session DB row (``sessions``), its event/message history, its memory stores
(``_memory_stores/`` — a SIBLING root under ``workspace_root``, never nested),
its attachments/uploads (``_attachments``/``_uploads`` — siblings), its snapshot
pointer, its outputs. ``rmtree`` of the workspace dir touches none of these.

The reap predicate (canonical-path, DB-driven)
----------------------------------------------
Unlike the #1192 reaper (filesystem-driven: scan dirs, parse the id off the
name), the workspace path is **DB-authoritative** (``sessions.workspace_volume_
path``) and may be a **user override** — the clone primitive explicitly lets two
sessions *share* one volume (``clone_session`` docstring). You cannot parse a
session id off such a path, and reaping a shared/overridden path would delete a
*different, possibly live* session's ``/workspace``.

So the reaper is DB-driven with a **canonical-path** confinement gate. For each
archived-and-aged session it derives the canonical default path
``<workspace_root>/<account_id>/<session_id>`` from the row's OWN id+account and
reaps **only** that, and **only** when the stored ``workspace_volume_path``
``resolve()``s equal to it. Consequences, all in the safe direction:

* A user-overridden / clone-shared / aliased / nested path never equals the
  canonical path ⇒ it is skipped (a space leak, never a wrong delete).
* The canonical path is always two levels under ``workspace_root`` ⇒ it can
  never be ``workspace_root`` itself or a reserved sibling root (``_runs`` …).
* A relative / empty / out-of-tree stored value never resolves-equal ⇒ skipped.

A symlink AT the canonical leaf is additionally never followed (its target may
be a live sibling session's dir, or escape the root). The symlink check runs on
the UN-resolved leaf — the path handed to ``rmtree`` is never symlink-
dereferenced — so a symlink swap can't redirect the delete onto a real victim.

Six guardrails (irreversible deletion — never-delete care)
----------------------------------------------------------
1. archived-only — ``archived_at IS NOT NULL`` (the DB query). A live/idle/
   running session is structurally excluded.
2. DB-liveness keep-set — the same query also requires ``NOT (active)`` using
   the wake sweep's predicate, read fresh at sweep time; a session with work
   still pending is never a candidate even if archived.
3. min-age floor — ``archived_at < now() - min_archived_age`` (DB time, default
   24h) plus a belt-and-suspenders dir-mtime floor (a dir touched recently —
   e.g. a step mid-write at archive — is left).
4. fail-closed — any uncertainty (DB error, unresolvable/mismatched path, stat
   failure, symlink) skips that item and deletes nothing on doubt. A DB error
   reaps nothing the whole sweep.
5. kill-switch — ``workspace_reaper_enabled`` (default OFF/dark): ships disabled,
   enabled after the seat's adversarial review. Disables the whole reaper with
   no redeploy.
6. dry-run + observability — ``workspace_reaper_dry_run`` logs what WOULD be
   reaped (+ reclaimable bytes) and deletes nothing; every sweep emits a
   structured ``ReapResult`` count for deploy verification.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.logging import get_logger

log = get_logger("aios.harness.workspace_reaper")

# Both ids are minted as ``<prefix>_<Crockford-ULID>`` (``aios.ids``), so a
# well-formed id is a non-empty run of ``[A-Za-z0-9_]`` only. We re-assert that
# here before interpolating an id into a delete-target path: the safety-relevant
# invariant is that the id can carry NO path separator, ``..``, NUL, leading
# ``.``, or whitespace — anything that could reshape ``<root>/<account>/<session>``
# (deepen it, or walk out of the root). A value off this shape is
# skipped-as-confinement, never reaped. Deliberately a character-class allowlist
# (not the exact ULID length) — it forecloses the path-escape class without
# coupling the reaper to the id minter's exact format.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ReapResult:
    """Structured per-sweep outcome — the observability surface (guardrail #6).

    ``reaped`` / ``bytes_freed`` are what was (or, in dry-run, WOULD be) deleted.
    The ``skipped_*`` counters partition every candidate that was *not* reaped,
    so a deploy can confirm the keep-set is doing its job:

    * ``skipped_confinement`` — stored path did not realpath-equal the canonical
      ``<root>/<account>/<session>`` path (off-shape id / symlink leaf / relative
      / out-of-tree / override / shared / aliased). The dominant safety skip.
    * ``skipped_missing`` — canonical dir not on disk (already gone / never
      provisioned). Benign.
    * ``skipped_too_fresh`` — dir mtime is under the mtime floor.
    * ``skipped_error`` — an ``rmtree`` raised (perm drift / read-only FS); the
      dir is left for the next sweep to retry. Kept distinct from ``missing`` so
      a recurring permission failure is visible, not masked as benign.
    """

    reaped: int = 0
    bytes_freed: int = 0
    skipped_confinement: int = 0
    skipped_missing: int = 0
    skipped_too_fresh: int = 0
    skipped_error: int = 0
    dry_run: bool = False


def _dir_size_bytes(path: Path) -> int:
    """Best-effort recursive on-disk size of ``path`` (apparent file sizes).

    Used only for the reclaimable-bytes counter, so an unreadable entry is
    skipped rather than fatal — the number is observability, not a gate.
    """
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_symlink() or not child.is_file():
                continue
            total += child.stat().st_size
        except OSError:
            continue
    return total


def _canonical_workspace_path(account_id: str, session_id: str, workspace_root: Path) -> Path:
    """The default per-session workspace dir ``<root>/<account>/<session>``.

    Recomputed from the row's own ids (never the stored string) so an
    override/shared/aliased ``workspace_volume_path`` can never be the reap
    target — that is the load-bearing confinement move.

    ``workspace_root`` must already be resolved by the caller. The leaf is NOT
    ``resolve()``d here: resolving would dereference a symlink AT the leaf, and
    the symlink check downstream must run on the un-dereferenced path (a symlink
    swap at ``<root>/<acct>/<sess>`` pointing at a live sibling's dir would
    otherwise be followed and reaped). This is the same un-resolved-leaf
    discipline ``host_dir_reaper._scan_children`` uses.
    """
    return workspace_root / account_id / session_id


def _resolve_reap_target(
    *,
    account_id: str,
    session_id: str,
    stored_path: str | None,
    workspace_root: Path,
    mtime_cutoff: float,
) -> tuple[Path | None, str]:
    """Decide whether (and where) to reap a single archived session's workspace.

    Returns ``(path, reason)``. ``path`` is the canonical dir to ``rmtree`` (NOT
    symlink-dereferenced), or ``None`` to skip; ``reason`` is one of ``"reap"`` /
    ``"confinement"`` / ``"missing"`` / ``"too_fresh"``. Every branch is
    fail-closed: anything we cannot positively confirm safe returns ``None``.

    ``workspace_root`` MUST be already-resolved (the caller resolves it once).
    """
    # Id-shape gate: a well-formed id can't contain a separator/``..``, so an
    # interpolated id can never reshape the path. Reject anything off-shape.
    if not (_ID_RE.match(account_id) and _ID_RE.match(session_id)):
        return None, "confinement"

    canonical = _canonical_workspace_path(account_id, session_id, workspace_root)

    # Never follow a symlink at the canonical leaf — its target may be a live
    # sibling session's dir or escape the root. Checked on the UN-resolved path
    # (``lstat``/``islink``), before any dereference, or this guard is dead.
    if canonical.is_symlink():
        return None, "confinement"

    # Confinement: reap ONLY the canonical default path, and ONLY when the
    # session's stored path points at the SAME real location. We compare on
    # ``realpath`` (symlink-collapsed) so an override / shared / aliased /
    # relative / out-of-tree stored value never matches ⇒ skipped, never reaped.
    if not stored_path:
        return None, "confinement"
    try:
        if os.path.realpath(stored_path) != os.path.realpath(canonical):
            return None, "confinement"
    except OSError:
        return None, "confinement"

    # Structural belt: a real dir that is exactly two levels under the root.
    if not canonical.is_dir():
        return None, "missing"
    if canonical.parent.parent != workspace_root:
        return None, "confinement"

    try:
        # lstat (not stat): mtime of the dir itself, never a dereferenced target
        # (the leaf is already proven non-symlink above; belt-and-suspenders).
        mtime = canonical.lstat().st_mtime
    except OSError:
        return None, "missing"
    if mtime > mtime_cutoff:
        return None, "too_fresh"

    return canonical, "reap"


async def sweep_archived_workspaces(pool: asyncpg.Pool[Any]) -> ReapResult:
    """One reaper sweep over archived-session ``/workspace`` dirs.

    Honors the kill-switch (``workspace_reaper_enabled``): disabled ⇒ deletes
    nothing, never queries the DB, returns an empty result. Fail-closed on a DB
    error: the candidate read is the only DB call, so a failure reaps nothing
    this pass. ``workspace_reaper_dry_run`` deletes nothing but still computes
    and logs the would-reap set + reclaimable bytes.
    """
    settings = get_settings()
    if not settings.workspace_reaper_enabled:
        return ReapResult()

    dry_run = bool(settings.workspace_reaper_dry_run)
    workspace_root = settings.workspace_root.resolve()
    mtime_cutoff = time.time() - float(settings.workspace_reaper_min_mtime_age_seconds)

    try:
        async with pool.acquire() as conn:
            candidates = await queries.unscoped_reapable_archived_workspaces(
                conn,
                min_archived_age_seconds=float(settings.workspace_reaper_min_archived_age_seconds),
            )
    except (asyncpg.PostgresError, OSError):
        # fail-closed: a DB hiccup must never be read as "everything is dead".
        log.exception("workspace_reaper.candidate_read_failed")
        return ReapResult()

    reaped = bytes_freed = skip_conf = skip_missing = skip_fresh = skip_error = 0
    for row in candidates:
        target, reason = _resolve_reap_target(
            account_id=row["account_id"],
            session_id=row["id"],
            stored_path=row["workspace_volume_path"],
            workspace_root=workspace_root,
            mtime_cutoff=mtime_cutoff,
        )
        if reason == "confinement":
            skip_conf += 1
            continue
        if reason == "missing":
            skip_missing += 1
            continue
        if reason == "too_fresh":
            skip_fresh += 1
            continue
        assert target is not None  # reason == "reap"

        size = _dir_size_bytes(target)
        if dry_run:
            reaped += 1
            bytes_freed += size
            log.info(
                "workspace_reaper.would_reap",
                session_id=row["id"],
                path=str(target),
                bytes=size,
            )
            continue
        try:
            shutil.rmtree(target)
        except OSError:
            # One un-removable dir (perm drift, read-only FS) must not abort the
            # rest of the sweep; the next sweep retries it.
            log.exception("workspace_reaper.rmtree_failed", path=str(target))
            skip_error += 1
            continue
        reaped += 1
        bytes_freed += size

    result = ReapResult(
        reaped=reaped,
        bytes_freed=bytes_freed,
        skipped_confinement=skip_conf,
        skipped_missing=skip_missing,
        skipped_too_fresh=skip_fresh,
        skipped_error=skip_error,
        dry_run=dry_run,
    )
    if candidates:
        log.info(
            "workspace_reaper.swept",
            reaped=result.reaped,
            bytes_freed=result.bytes_freed,
            skipped_confinement=result.skipped_confinement,
            skipped_missing=result.skipped_missing,
            skipped_too_fresh=result.skipped_too_fresh,
            skipped_error=result.skipped_error,
            dry_run=result.dry_run,
            candidates=len(candidates),
        )
    return result
