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

* A user-overridden / clone-shared / nested path that resolves elsewhere never
  equals the canonical path ⇒ it is skipped (a space leak, never a wrong delete).
  The candidate query already leaves out every row whose stored path is not
  spelled exactly ``<configured root>/<account_id>/<id>``. When the configured
  root is a symlink, that spelling is an alias of the canonical path and passes
  this check; the canonical delete target itself must still contain no symlink
  component.
* The canonical path is always two levels under ``workspace_root`` ⇒ it can
  never be ``workspace_root`` itself or a reserved sibling root (``_runs`` …).
* Empty, relative, or out-of-tree stored values are rejected explicitly.

Confinement is proven on the FULLY-RESOLVED canonical path — no symlink anywhere
in its chain (leaf, parent ACCOUNT component, OR root) and its realpath still
exactly ``<resolved_root>/<account>/<session>``. A symlink at ANY component
(e.g. a swapped ``<root>/<account>`` redirecting at another account's LIVE dir)
breaks that and is skipped. The path handed to ``rmtree`` is the UN-resolved
canonical — never symlink-dereferenced — so a swap can't redirect the delete
onto a real victim. (The CHECK resolves; the TARGET does not — they are kept
deliberately separate, since re-merging them is what caused the prior bugs.)

A residual same-class guard: a LIVE clone can share an archived parent's OWN
canonical default dir (``clone_session`` shares volumes). Before reaping, the
candidate's canonical realpath is cross-checked against the keep-set of every
non-archived session's workspace realpath; a collision skips the candidate so
reaping the parent never deletes a live clone's volume.

Candidates and cost
-------------------
The candidate query returns only sessions still on their default path, the
one directory this reaper deletes (``unscoped_reapable_archived_workspaces``
lists what that leaves out). The filesystem work runs in a worker thread, so a
pass never blocks the event loop (aios#2581).

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

import asyncio
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
from aios.db.pool import normalize_dsn
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
# coupling the reaper to the id minter's exact format. Always ``fullmatch`` it:
# an anchored ``$`` pattern also matches an id ending in "\n".
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class ReapResult:
    """Structured per-sweep outcome — the observability surface (guardrail #6).

    ``reaped`` / ``bytes_freed`` are what was (or, in dry-run, WOULD be) deleted.
    The ``skipped_*`` counters partition every candidate that was *not* reaped,
    so a deploy can confirm the keep-set is doing its job:

    * ``skipped_confinement`` — off-shape id, a symlink in the canonical chain,
      a stored path that does not realpath-equal the canonical
      ``<root>/<account>/<session>`` path, or a live workspace overlapping it.
      Rows whose stored path is not their default spelling never get here: the
      candidate query leaves them out, so this counter does not include them.
    * ``skipped_missing`` — canonical dir not on disk (already gone / never
      provisioned). Benign.
    * ``skipped_too_fresh`` — dir mtime is under the mtime floor.
    * ``skipped_error`` — a filesystem check or the ``rmtree`` raised (perm
      drift / read-only FS); the dir is left for the next sweep to retry. Kept
      distinct from ``missing`` so a recurring permission failure is visible,
      not masked as benign.
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


def _self_and_ancestors(path: str) -> list[str]:
    """``path`` followed by every directory above it, up to ``/``."""
    return [path, *map(str, Path(path).parents)]


@dataclass(frozen=True, slots=True)
class _LiveWorkspaces:
    """The realpaths of the workspaces that live sessions and shared runs use.

    ``paths`` holds each live realpath. ``covered`` holds each live realpath
    and every directory above it, so a directory that contains a live
    workspace is a member, and an overlap check is a few set lookups.
    """

    paths: frozenset[str]
    covered: frozenset[str]

    def overlaps(self, path: str) -> bool:
        """Whether a live workspace is ``path``, lies inside it, or contains it."""
        return path in self.covered or not self.paths.isdisjoint(_self_and_ancestors(path))


def _raise(err: OSError) -> None:
    raise err


def _max_tree_mtime(path: Path) -> float:
    """Newest lstat mtime in a workspace tree, without following symlinks.

    A directory that cannot be listed raises instead of being skipped, so an
    unreadable subtree can never hide a recent write from the mtime floor.
    """
    newest = path.lstat().st_mtime
    for root, dirs, files in os.walk(path, followlinks=False, onerror=_raise):
        for name in (*dirs, *files):
            newest = max(newest, (Path(root) / name).lstat().st_mtime)
    return newest


def _live_workspace_realpath_keepset(live_paths: list[str]) -> _LiveWorkspaces:
    """Realpath-normalize the live-clone keep-set (Fix 2), in a sync helper.

    Non-strict ``realpath`` resolves a missing path as far as it exists rather
    than raising. If it does raise, the error propagates and the pass reaps
    nothing: an incomplete keep-set must never be treated as "no live clones".
    """
    out = {os.path.realpath(p) for p in live_paths}
    return _LiveWorkspaces(
        paths=frozenset(out),
        covered=frozenset(a for p in out for a in _self_and_ancestors(p)),
    )


def _resolve_reap_target(
    *,
    account_id: str,
    session_id: str,
    stored_path: str,
    workspace_root: Path,
    mtime_cutoff: float,
    live_workspaces: _LiveWorkspaces,
) -> tuple[Path | None, str]:
    """Decide whether (and where) to reap a single archived session's workspace.

    Returns ``(path, reason)``. ``path`` is the canonical dir to ``rmtree`` (NOT
    symlink-dereferenced), or ``None`` to skip; ``reason`` is one of ``"reap"`` /
    ``"confinement"`` / ``"missing"`` / ``"too_fresh"``. Every branch is
    fail-closed: anything we cannot positively confirm safe returns ``None``,
    and a directory check or tree walk that fails with anything but a missing
    path (a permission error, say) raises ``OSError``, which the caller counts
    as ``skipped_error``.

    ``workspace_root`` MUST be already-resolved (the caller resolves it once).
    ``live_workspaces`` is the realpath keep-set of every LIVE (non-archived)
    session's stored workspace path (the caller builds it once); a candidate
    whose canonical realpath collides with a live path is skipped (a live clone
    can share an archived parent's own canonical dir).
    """
    # Id-shape gate: a well-formed id can't contain a separator/``..``, so an
    # interpolated id can never reshape the path. Reject anything off-shape.
    if not (_ID_RE.fullmatch(account_id) and _ID_RE.fullmatch(session_id)):
        return None, "confinement"

    canonical = _canonical_workspace_path(account_id, session_id, workspace_root)

    # ── Confinement CHECK vs. rmtree TARGET — two DIFFERENT resolutions ──────
    # These two concerns MUST stay separate (re-merging them is what caused the
    # two prior confinement bugs). The CHECK proves safety on the FULLY-RESOLVED
    # path (symlinks collapsed at EVERY component — leaf, parent, AND root); the
    # TARGET handed to ``rmtree`` is the UN-resolved ``canonical`` (a path with a
    # symlink anywhere in its chain must never be dereferenced into a delete).
    #
    # Why the earlier guards were insufficient (the cross-account LIVE-data
    # delete the review proved): a symlink at the ACCOUNT component
    # ``<root>/<account_id>`` is invisible to a leaf-only ``is_symlink()`` check,
    # and ``realpath(stored) == realpath(canonical)`` collapses that parent
    # symlink IDENTICALLY on both sides (so it matches and waves the candidate
    # through), and ``canonical.parent.parent == workspace_root`` is computed on
    # the UN-resolved path (so it passes trivially). ``rmtree(canonical)`` then
    # follows the parent symlink and deletes its real target — which can be
    # ANOTHER account's LIVE session dir. ``volumes.py`` documents in-sandbox
    # symlink swaps under ``workspace_root`` as a real, defended-against threat,
    # so this is reachable, not theoretical.
    #
    # The fix: confirm the canonical path has NO symlink anywhere in its chain
    # (``realpath(canonical) == canonical``). ``canonical`` is the RESOLVED root
    # joined with two well-formed ids, so that one equality also proves its
    # realpath is exactly ``<resolved root>/<account_id>/<session_id>``: a
    # symlink at ANY component (leaf OR parent OR root, including a root that
    # became a symlink after the caller resolved it) makes the realpath differ
    # from the un-resolved string ⇒ skipped, never reaped.
    #
    # IMPORTANT: never replace ``canonical`` (the rmtree target) with its
    # ``.resolve()``/``realpath`` form — the target must stay the un-resolved
    # path so a swap can't redirect the delete onto a real victim.
    try:
        canonical_real = os.path.realpath(canonical)
    except OSError:
        return None, "confinement"
    if canonical_real != str(canonical):
        # A symlink exists somewhere in the canonical chain (leaf, parent, or
        # root). The un-resolved path differs from its realpath ⇒ fail closed.
        return None, "confinement"

    # Belt-and-suspenders: never follow a symlink at the canonical leaf either.
    # Checked on the UN-resolved path (``lstat``/``islink``); redundant with the
    # realpath-equality above but kept as a cheap, explicit leaf guard so a
    # future refactor of the chain check can't silently reintroduce leaf-follow.
    if canonical.is_symlink():
        return None, "confinement"

    # Confinement: reap ONLY the canonical default path, and ONLY when the
    # session's stored path points at the SAME real location. Relative paths are
    # rejected before realpath so the decision never depends on process CWD.
    if not os.path.isabs(stored_path):
        return None, "confinement"
    try:
        if os.path.realpath(stored_path) != canonical_real:
            return None, "confinement"
    except OSError:
        return None, "confinement"

    # Live-clone keep-set cross-check (same never-delete class): a LIVE clone can
    # share an archived parent's OWN canonical default dir. Reaping the archived
    # parent's row would ``rmtree`` the directory the live clone is still using ⇒
    # cross-session live data loss. Skip if this candidate's canonical realpath
    # collides with any live session's workspace realpath. (The keep-set is
    # fail-closed-empty only when its query erred — and on that error the caller
    # skips the WHOLE sweep, so an empty set here always means "no live collision
    # among the sessions we could enumerate".)
    if live_workspaces.overlaps(canonical_real):
        return None, "confinement"

    if not canonical.is_dir():
        return None, "missing"

    try:
        # Descendant writes do not update the candidate directory's mtime, so
        # inspect the full tree. os.walk never follows symlinked directories.
        mtime = _max_tree_mtime(canonical)
    except FileNotFoundError:
        return None, "missing"
    if mtime > mtime_cutoff:
        return None, "too_fresh"

    return canonical, "reap"


async def sweep_archived_workspaces(pool: asyncpg.Pool[Any]) -> ReapResult:
    """One reaper sweep over archived-session ``/workspace`` dirs.

    Honors the kill-switch (``workspace_reaper_enabled``): disabled ⇒ deletes
    nothing, never queries the DB, returns an empty result. Fail-closed on a DB
    error: if either the candidate read or the live keep-set read fails, the
    pass reaps nothing. ``workspace_reaper_dry_run`` deletes nothing but still
    computes and logs the would-reap set + reclaimable bytes.

    Classification, the keep-set's realpaths and the reclaimable-size walk run
    in a worker thread, so the event loop keeps serving session steps during a
    pass (aios#2581).
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
                # The configured root, not the resolved one: stored default
                # paths are spelled from it.
                workspace_root=settings.workspace_root,
            )
            # Live-clone keep-set (Fix 2): every non-archived session's workspace
            # path, realpath-normalized. Read AFTER the candidate set (each
            # statement takes its own READ COMMITTED snapshot), so the keep-set
            # is at least as fresh as the candidates it protects against.
            live_paths = await queries.unscoped_live_workspace_volume_paths(conn)
    except (asyncpg.PostgresError, OSError):
        # fail-closed: a DB hiccup must never be read as "everything is dead".
        # If EITHER read fails (candidates or live keep-set) we reap nothing this
        # pass — an absent keep-set must never be treated as "no live clones".
        log.exception("workspace_reaper.candidate_read_failed")
        return ReapResult()

    live_workspaces = await asyncio.to_thread(_live_workspace_realpath_keepset, live_paths)

    reaped = bytes_freed = skip_conf = skip_missing = skip_fresh = skip_error = 0
    # One non-pooled backend is dedicated to this sweep's lock lifecycle.  The
    # session advisory lock must remain owned across slow filesystem deletion;
    # borrowing that backend from the shared pool would starve unrelated work,
    # while opening one backend per candidate would create a connection storm.
    lock_conn = await asyncpg.connect(
        normalize_dsn(settings.db_url),
        server_settings={"application_name": "aios-workspace-reaper"},
    )
    try:
        for row in candidates:
            try:
                target, reason = await asyncio.to_thread(
                    _resolve_reap_target,
                    account_id=row["account_id"],
                    session_id=row["id"],
                    stored_path=row["workspace_volume_path"],
                    workspace_root=workspace_root,
                    mtime_cutoff=mtime_cutoff,
                    live_workspaces=live_workspaces,
                )
            except OSError:
                # An unreadable path (perm drift) skips this candidate only:
                # nothing is deleted, and the next sweep retries it.
                log.exception("workspace_reaper.classify_failed", session_id=row["id"])
                skip_error += 1
                continue
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

            size = await asyncio.to_thread(_dir_size_bytes, target)
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
                # The dedicated sweep backend owns the session lock across slow
                # filesystem I/O; explicit unlock makes it reusable for the next item.
                normalized_target = queries.normalized_workspace_path(str(target))
                lock_key = queries.workspace_advisory_lock_key(normalized_target)
                await lock_conn.execute("SELECT pg_advisory_lock($1::bigint)", lock_key)
                try:
                    if await queries.unscoped_workspace_path_is_live(lock_conn, normalized_target):
                        skip_conf += 1
                        continue
                    await asyncio.to_thread(shutil.rmtree, target)
                finally:
                    await lock_conn.execute("SELECT pg_advisory_unlock($1::bigint)", lock_key)
            except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError):
                # One un-removable dir (perm drift, read-only FS) must not abort the
                # rest of the sweep; the next sweep retries it.
                log.exception("workspace_reaper.rmtree_failed", path=str(target))
                skip_error += 1
                continue
            reaped += 1
            bytes_freed += size
    finally:
        await lock_conn.close()
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
