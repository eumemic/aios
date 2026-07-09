"""Post-exec memory-mount reconciliation for the bash tool.

After each bash command completes, we diff the memory-mount host directories
against a snapshot taken before the command ran. Any files that were created,
modified, or deleted by the bash command are propagated to the database as
``memory_versions`` rows, closing the v2 limitation where bash writes were
visible cross-session via the shared FS but did not produce durable version
history.

Design constraints:
- Snapshot is a cheap ``os.stat`` walk (no bytes read, no hash) — taken before
  ``sandbox.exec``, off the event loop via ``asyncio.to_thread``.
- Reconcile is async — runs after ``sandbox.exec`` returns; its walk/stat/read
  phase also runs off the loop via ``asyncio.to_thread``, only the DB calls
  stay on the loop.
- read_only mounts are never written; we skip them in both directions.
- Binary files and oversized files emit warnings instead of failing loudly
  (the command already ran; we just can't record what it wrote).
- sha256 is over UTF-8 encoded bytes of the content string, matching the
  convention in :mod:`aios.services.memory_stores._sha256_hex`.

Stat-signature prefilter (#1748)
---------------------------------
Master (pre-#1748) read + sha256-hashed every file in every writable memory
store, TWICE per bash call, synchronously on the event loop. That is exactly
the on-loop-hash pattern #1733 (zero-inference-gap) forbids, and it scales
with total memory-store bytes rather than with what the command actually
touched.

The fix replaces content hashing with a cheap ``os.stat`` 4-tuple signature
``(st_size, st_mtime_ns, st_ctime_ns, st_ino)`` for the before/after walks.
**``st_ctime_ns`` is the ONLY correctness anchor** in that tuple — the kernel
bumps it on every inode data change and unprivileged code cannot backdate it
(no ``CAP_SYS_TIME``, no raw-device access inside the sandbox). ``st_size``,
``st_mtime_ns``, and ``st_ino`` are free redundancy riding the same
``stat()`` syscall: they can only ever ADD candidates (the safe direction),
never remove one that ctime alone would have caught. Do not "optimize" this
later by trusting size+mtime instead of ctime — a `touch -r`/`cp -p` mtime
backdate would then silently miss a real content change.

Hot-window (why it is load-bearing, not belt-and-suspenders): ctime is
*stored* at nanosecond granularity on ext4/xfs, but it is *sourced* from the
kernel's COARSE realtime clock (``ktime_get_coarse_real_ts64``), which ticks
only once per timer granule (~1-10ms on stock kernels). Two writes inside the
same tick can therefore observe an IDENTICAL ``st_ctime_ns`` even on an
ns-granular filesystem. Any file whose before-observed ctime falls within
``HOT_WINDOW_NS`` of the pre-exec snapshot timestamp is unconditionally
treated as a candidate regardless of what the after-stat says — this is what
actually closes the same-tick collision, not filesystem storage granularity.
Never shrink ``HOT_WINDOW_NS`` toward "the ns granule": that reopens exactly
the silent-loss hole it exists to close.

Fail-closed coarse/absent-ctime guard: a file whose ``stat()`` raises, or
whose ``st_ctime_ns == 0`` (a filesystem without ctime), gets a sentinel sig
that never compares equal to anything (including itself across calls) — so
it is always a candidate. A per-mount runtime probe (see
:func:`probe_mount_ctime_granularity`) additionally disables the whole
prefilter (forcing every candidate to be hashed) if the observed ctime
granule on the actual memory-store mount is at or above ``HOT_WINDOW_NS``,
because that mount is slated to potentially move onto overlayfs/tmpfs/NFS
underfoot.

Accepted residual: a detached process writing via a shared ``mmap`` can defer
its timestamp update to writeback. If that writeback lands strictly BETWEEN
two bash calls (no call's [before, after] window straddles it), the next
call's before-snapshot already reflects the post-writeback state and, if that
call doesn't otherwise touch the file, the write may be PERMANENTLY
unreconciled (not merely deferred) — identical to master's blind spot for a
single command's before/after window, and outside the supported
memory-write pattern (mmap'd background writers), so this is a bounded,
documented, non-regressing residual, not a new hole.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from aios.harness import runtime
from aios.logging import get_logger
from aios.models.memory_stores import MAX_CONTENT_BYTES
from aios.sandbox.memory_mounts import MATERIALIZED_MARKER
from aios.sandbox.volumes import memory_store_host_dir
from aios.services import memory_stores as memory_service
from aios.services import sessions as sessions_service
from aios.services.memory_stores import SessionActor
from aios.tools import bash_memory_telemetry as telemetry

log = get_logger("aios.tools.bash_memory_reconcile")

# Hot-window: any file whose before-ctime is within this many nanoseconds of
# the pre-exec snapshot timestamp is ALWAYS a candidate, regardless of the
# after-stat comparison. See module docstring — this is the load-bearing
# catch for same-coarse-tick ctime collisions, not a redundant belt.
HOT_WINDOW_NS = 2_000_000_000  # 2 seconds


@dataclass(frozen=True, slots=True)
class _Sig:
    """A stat-derived change-detection signature for one file.

    ``ctime_ns`` is the only field this module's correctness argument rests
    on. ``size``, ``mtime_ns``, and ``ino`` are free redundancy from the same
    ``stat()`` call: including them in the equality check can only ever make
    MORE files candidates than ctime alone would (the safe direction), never
    fewer. Do not remove ctime in favor of the other three — see module
    docstring.
    """

    size: int
    mtime_ns: int
    ctime_ns: int
    ino: int


# A sentinel signature that never compares equal to anything, including
# another instance of itself (via NaN-like identity semantics using a
# monotonically-unique marker). Used for stat-failures and absent-ctime
# filesystems so such paths are ALWAYS treated as candidates — the fail-safe
# direction on the axis where a false negative loses memory.
class _NeverEqualSig(_Sig):
    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:  # pragma: no cover - dict usage doesn't rely on this
        return id(self)


_SENTINEL_SIG = _NeverEqualSig(size=-1, mtime_ns=-1, ctime_ns=-1, ino=-1)

# Snapshot type: (store_id, store_path) -> stat signature.
_Snapshot = dict[tuple[str, str], _Sig]


class _Unreadable:
    """Zero-field sentinel marking a walked file whose ``read_bytes()`` raised.

    Stored in the candidate-bytes map in place of bytes so the path stays
    PRESENT in the after-scan. This keeps the diff from mistaking "unreadable"
    for "absent" and reconciling it as a deletion (which would wipe the
    file's version history while it is still on disk).
    """


# Module-singleton sentinel for unreadable files.
UNREADABLE = _Unreadable()

# Type of the candidate-bytes map value: real bytes, or the unreadable sentinel.
_BytesMap = dict[tuple[str, str], "bytes | _Unreadable"]


def _stat_sig(fpath: Path) -> _Sig:
    """Return the stat-signature for ``fpath``, or the never-equal sentinel.

    Fail-closed: any ``OSError`` from ``stat()`` (permission, race with an
    in-flight delete, transient FS hiccup) or an absent-ctime filesystem
    (``st_ctime_ns == 0``) routes to the sentinel so the path is unconditionally
    a candidate rather than silently compared as "unchanged".
    """
    try:
        st = os.lstat(fpath)
    except OSError:
        return _SENTINEL_SIG
    ctime_ns = st.st_ctime_ns
    if ctime_ns == 0:
        return _SENTINEL_SIG
    return _Sig(size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=ctime_ns, ino=st.st_ino)


def _walk_store_files(host_dir: Path) -> list[tuple[Path, str]]:
    """Return (absolute_path, store_path) for all eligible files in host_dir.

    Skips directories, the .materialized marker, any hidden temp files
    starting with ``.tmp.`` (written by the atomic_mirror helper during
    in-flight writes), and any symlinks. The symlink rejection mirrors
    PR #497's policy for ``walk_skill_dir``: bash inside the sandbox can
    ``ln -s <worker-side-path> /mnt/memory/<store>/leak``, and the
    subsequent host-side read would resolve the symlink against the
    worker's filesystem — exfiltrating any worker-readable file (vault key
    material on disk, neighbour tenants' state, /etc, OAuth refresh
    tokens, …) into a memory store that then persists the bytes to DB and
    renders them to the model on every wake.

    LOAD-BEARING — preserved verbatim from pre-#1748 (#1705 depends on this
    walk's exact skip semantics).
    """
    results: list[tuple[Path, str]] = []
    for fpath in host_dir.rglob("*"):
        if fpath.is_symlink():
            continue
        if fpath.is_dir():
            continue
        name = fpath.name
        if name == MATERIALIZED_MARKER:
            continue
        if name.startswith(".tmp."):
            continue
        rel = fpath.relative_to(host_dir)
        store_path = "/" + str(rel).replace("\\", "/")
        results.append((fpath, store_path))
    return results


def has_writable_memory_mount(session_id: str) -> bool:
    """True if the session has at least one non-``read_only`` memory-store echo.

    Cheap: ``runtime.get_session_memory_mounts`` is a plain in-process dict
    read, no I/O. Used by both ``bash.py`` (to decide whether to even
    dispatch the before-snapshot ``to_thread`` hop) and this module's own
    fast path (to skip the after-scan ``to_thread`` hop) — #1748 step 5.
    A session with only read_only mounts (or none at all) can never produce
    a create/modify/delete, so both hops are pure overhead for it.
    """
    return any(echo.access != "read_only" for echo in runtime.get_session_memory_mounts(session_id))


def _iter_scannable_stores(
    session_id: str,
) -> list[tuple[str, Path]]:
    """Yield ``(store_id, host_dir)`` for every writable, materialized mount.

    Shared gate logic between the before-snapshot and the after-scan: a
    writable echo whose host dir exists and carries the ``.materialized``
    marker. LOAD-BEARING — preserved verbatim from pre-#1748.
    """
    echoes = runtime.get_session_memory_mounts(session_id)
    result: list[tuple[str, Path]] = []
    for echo in echoes:
        if echo.access == "read_only":
            continue
        store_id = echo.memory_store_id
        host_dir = memory_store_host_dir(store_id)
        if not host_dir.exists():
            continue
        marker = host_dir / MATERIALIZED_MARKER
        if not marker.exists():
            continue
        result.append((store_id, host_dir))
    return result


def _snapshot_sigs(session_id: str) -> _Snapshot:
    """Stat-only walk of every writable, materialized memory mount.

    No bytes read, no hashing — one ``os.stat`` per file. Sync/cheap enough
    to run inline in the caller's thread (callers off the event loop via
    ``asyncio.to_thread``).
    """
    snapshot: _Snapshot = {}
    for store_id, host_dir in _iter_scannable_stores(session_id):
        for fpath, store_path in _walk_store_files(host_dir):
            snapshot[(store_id, store_path)] = _stat_sig(fpath)
    return snapshot


def snapshot_memory_mounts(session_id: str) -> tuple[_Snapshot, int]:
    """Return a stat-signature snapshot plus the pre-exec timestamp.

    Called before ``sandbox.exec`` (via ``asyncio.to_thread`` from
    ``bash.py`` — this function itself is plain sync code, safe to run in a
    worker thread). Returns ``(before_sigs, snapshot_ns)``:

    - ``before_sigs``: ``(store_id, store_path) -> _Sig`` for every eligible
      file, no bytes read, no hash.
    - ``snapshot_ns``: ``time.time_ns()`` sampled HERE, pre-exec. This is a
      hard contract (Lens 0 #2): the after-scan MUST be given this same
      value, never re-sample it. Sampling post-exec would make every file
      whose ctime predates ``snapshot_ns`` by less than one hot-window look
      "not hot" even if it was modified moments before a long-running exec —
      silently disabling the hot-window net for any command that runs longer
      than ``HOT_WINDOW_NS`` (routine: default bash timeout is 120s).

    Skips:
    - Sessions with no attached stores.
    - read_only mounts.
    - Stores whose host directory does not exist yet.
    - Stores that have not been materialized (marker file absent).
    """
    snapshot_ns = time.time_ns()
    sigs = _snapshot_sigs(session_id)
    return sigs, snapshot_ns


def _is_hot(sig: _Sig, snapshot_ns: int) -> bool:
    """True if ``sig``'s ctime falls within ``HOT_WINDOW_NS`` of the snapshot.

    The sentinel sig (ctime_ns=-1) is never "hot" by this check alone — it's
    already a candidate unconditionally via the sentinel's never-equal
    semantics, so hot-window membership is moot for it.
    """
    if sig is _SENTINEL_SIG:
        return False
    return sig.ctime_ns >= snapshot_ns - HOT_WINDOW_NS


def _is_candidate(
    before_sig: _Sig | None,
    after_sig: _Sig,
    snapshot_ns: int,
    *,
    force_hash: bool,
) -> bool:
    """True if this path must be read+hashed rather than trusted as unchanged.

    A path is skipped (NOT a candidate) only if: it existed in ``before``
    AND both sigs are non-sentinel AND they compare equal AND it is not hot.
    Everything else — new path, missing before entry, sentinel on either
    side, sig-differs, hot, or the coarse-ctime force-hash escape armed —
    is a candidate.
    """
    if force_hash:
        return True
    if before_sig is None:
        return True  # new path
    if before_sig is _SENTINEL_SIG or after_sig is _SENTINEL_SIG:
        return True
    if before_sig != after_sig:
        return True
    return _is_hot(before_sig, snapshot_ns)


def _read_candidate(fpath: Path) -> bytes | _Unreadable:
    """Read a candidate file's bytes, immediately after the symlink check.

    Mirrors the pre-#1748 ``_snapshot_with_bytes`` read: happens in the SAME
    walk iteration as the symlink rejection, never via a later re-open by
    reconstructed path (which could race a symlink planted by a peer
    session between the walk and a deferred read).
    """
    try:
        return fpath.read_bytes()
    except OSError:
        return UNREADABLE


@dataclass(frozen=True, slots=True)
class _ScanResult:
    after_sigs: _Snapshot
    candidate_bytes: _BytesMap
    scanned_store_ids: set[str]
    candidate_read_count: int


def _scan_after(
    session_id: str,
    before: _Snapshot,
    snapshot_ns: int,
    *,
    force_hash: bool,
) -> _ScanResult:
    """Re-walk every store; stat every file; read bytes for candidates only.

    Hard invariant: ``after_sigs`` contains EVERY walked path, including
    paths whose ``os.stat`` raised (recorded under the sentinel sig) and
    paths whose candidate read later fails. The delete-diff in
    :func:`reconcile_memory_mounts` keys off ``after_sigs`` presence, NEVER
    off the (smaller) candidate-bytes set — a builder who populated
    ``after_sigs`` only for "valid, comparable" sigs would let a transient
    ``os.stat`` failure on an untouched, unrelated file read as "absent" and
    reconcile it as a delete (the #1705 mass-delete class).

    Runs entirely off the event loop (call via ``asyncio.to_thread``).
    """
    after_sigs: _Snapshot = {}
    candidate_bytes: _BytesMap = {}
    scanned_store_ids: set[str] = set()
    candidate_read_count = 0

    for store_id, host_dir in _iter_scannable_stores(session_id):
        scanned_store_ids.add(store_id)
        for fpath, store_path in _walk_store_files(host_dir):
            after_sig = _stat_sig(fpath)
            after_sigs[(store_id, store_path)] = after_sig

            before_sig = before.get((store_id, store_path))
            if _is_candidate(before_sig, after_sig, snapshot_ns, force_hash=force_hash):
                candidate_read_count += 1
                candidate_bytes[(store_id, store_path)] = _read_candidate(fpath)

    return _ScanResult(
        after_sigs=after_sigs,
        candidate_bytes=candidate_bytes,
        scanned_store_ids=scanned_store_ids,
        candidate_read_count=candidate_read_count,
    )


def _content_sha256(raw: bytes) -> str:
    """sha256 over UTF-8-decoded content when possible, else over raw bytes.

    Mirrors :mod:`aios.services.memory_stores`' ``_sha256_hex`` convention
    (sha over the decoded string), falling back to raw bytes for binary
    blobs so a change can still be detected even though it can't be stored.
    """
    try:
        content = raw.decode("utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except UnicodeDecodeError:
        return hashlib.sha256(raw).hexdigest()


def _read_utf8_content(
    store_id: str,
    store_path: str,
    warnings: list[str],
    session_id: str,
    raw: bytes,
) -> str | None:
    """Decode ``raw`` as UTF-8 and enforce the size limit.

    Returns the decoded string on success, or ``None`` if the bytes cannot be
    decoded or exceed ``MAX_CONTENT_BYTES``. In the rejection case a human-
    readable entry is appended to ``warnings`` and a structured warning is
    emitted via the logger.
    """
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        warnings.append(
            f"skipped {store_path!r} in store {store_id}: binary file cannot be stored as memory"
        )
        log.warning(
            "memory_reconcile.binary_file_skipped",
            session_id=session_id,
            store_id=store_id,
            store_path=store_path,
        )
        return None
    if len(raw) > MAX_CONTENT_BYTES:
        warnings.append(
            f"skipped {store_path!r} in store {store_id}: "
            f"file size {len(raw)} exceeds {MAX_CONTENT_BYTES}-byte limit"
        )
        log.warning(
            "memory_reconcile.oversized_file_skipped",
            session_id=session_id,
            store_id=store_id,
            store_path=store_path,
            size=len(raw),
        )
        return None
    return content


def _warn_unreadable(
    store_id: str,
    store_path: str,
    warnings: list[str],
    session_id: str,
) -> None:
    """Surface an unreadable file via the warnings channel and a log line.

    Mirrors :func:`_read_utf8_content`'s structure: an unreadable file is the
    third skipped class alongside binary and oversized files, surfaced instead
    of being acted on as a create/update/delete.
    """
    warnings.append(
        f"skipped {store_path!r} in store {store_id}: file became unreadable; not reconciled"
    )
    log.warning(
        "memory_reconcile.unreadable_file_skipped",
        session_id=session_id,
        store_id=store_id,
        store_path=store_path,
    )


def _warn_store_unscannable(
    store_id: str,
    suppressed_deletes: int,
    warnings: list[str],
    session_id: str,
) -> None:
    """Surface a store that dropped out of the after-scan; suppress its deletes.

    The store-level twin of :func:`_warn_unreadable` (#1705). A store present
    in ``before`` but absent from the after-pass ``scanned_store_ids`` was not
    actually observed after ``sandbox.exec`` — its mount echo-cache was cleared
    mid-bash (``SandboxRegistry.release``/``evict``/idle-reaper), or its host dir
    or ``.materialized`` marker vanished. Its ``before`` paths therefore read as
    "deleted" only because the after-view is UNKNOWN, not because the files were
    removed (they are still on disk and in the DB). Deleting on that view is the
    mass-delete data-loss path, so every delete for the store is suppressed and
    the store is surfaced instead — emitted once per store, not once per file.

    LOAD-BEARING — must run regardless of whether ``account_id`` has been
    loaded (it needs none); see the ordering hazard note in
    :func:`reconcile_memory_mounts`.
    """
    warnings.append(
        f"store {store_id}: not scannable after exec (mount cache cleared or "
        f"host dir/marker absent); suppressed {suppressed_deletes} deletion(s) "
        f"to avoid data loss"
    )
    log.warning(
        "memory_reconcile.store_unscannable",
        session_id=session_id,
        store_id=store_id,
        suppressed_deletes=suppressed_deletes,
    )


async def reconcile_memory_mounts(
    session_id: str, before: _Snapshot, snapshot_ns: int
) -> list[str]:
    """Diff memory mount state against ``before``; write DB changes.

    Called after ``sandbox.exec`` returns. For each writable mount:

    - New files (in after, not in before): ``create_memory``
    - Modified candidates (sig differs from a valid DB sha comparison):
      ``update_memory`` with precondition
    - Deleted files (in before, not in after_sigs): ``delete_memory``
    - Unchanged files (not a candidate): no read, no DB call

    The walk/stat/candidate-read phase runs off the event loop via
    ``asyncio.to_thread``; only the DB phase below stays on the loop.

    Fast path (#1748 step 5): if there is nothing to reconcile — zero
    candidates AND zero real deletes — ``load_session_account_id`` is never
    called. "Zero real deletes" is evaluated AFTER the #1705 unscannable-store
    suppression classification: an unscannable store's suppressed deletes are
    NOT "real" deletes for this purpose (nothing gets written to the DB for
    them), so a no-op call against an otherwise-unscannable store still hits
    the fast path — but the suppression warning is emitted regardless,
    UNCONDITIONALLY on whether the fast path fires, because it needs no
    ``account_id`` and is the #1705 regression guard's whole point.

    Binary files and files exceeding ``MAX_CONTENT_BYTES`` are skipped with a
    warning string (collected and returned). DB errors propagate — they are
    the session's problem to recover from through the normal error channel.
    """
    # Fast path (#1748 step 5, bash.py-mirrored): a session with no attached
    # memory-store echoes at all (not even read_only ones) cannot have
    # anything to scan — ``_iter_scannable_stores`` would yield nothing for
    # either pass. Skip the ``to_thread`` dispatch entirely rather than
    # paying a thread-pool round trip just to discover an empty result.
    # ``runtime.get_session_memory_mounts`` is a plain dict read (no I/O), so
    # this check is free.
    if not before and not has_writable_memory_mount(session_id):
        telemetry.record_candidate_reads(0)
        return []

    force_hash = not get_prefilter_state().enabled

    with telemetry.timed_phase("memory_reconcile"):
        with telemetry.timed_phase("after_scan"):
            scan = await asyncio.to_thread(
                _scan_after, session_id, before, snapshot_ns, force_hash=force_hash
            )

        after_sigs = scan.after_sigs
        candidate_bytes = scan.candidate_bytes
        after_scanned_stores = scan.scanned_store_ids
        telemetry.record_candidate_reads(scan.candidate_read_count)

        warnings: list[str] = []

        # ── Deleted files: classify FIRST (no account_id needed) ────────────
        # Count before-paths per store so an unscannable store can report how
        # many deletions it suppressed (the #1705 incident wrote 159 durable
        # deletes in 1.79s from a read-only bash while the mount cache was
        # cleared under it).
        before_path_counts = Counter(store_id for store_id, _store_path in before)
        unscannable_warned: set[str] = set()
        # (store_id, store_path) pairs that are genuinely deleted (store WAS
        # scanned, path is absent from after_sigs) — these are the only
        # deletes that will actually hit the DB.
        real_deletes: list[tuple[str, str]] = []
        for store_id, store_path in before:
            if (store_id, store_path) in after_sigs:
                continue  # still present
            if store_id not in after_scanned_stores:
                # Store not observed in the after-pass — before-paths are
                # UNKNOWN, not deleted. Warn once per store; suppress the
                # delete. Runs regardless of the fast-path / account_id state.
                if store_id not in unscannable_warned:
                    unscannable_warned.add(store_id)
                    _warn_store_unscannable(
                        store_id, before_path_counts[store_id], warnings, session_id
                    )
                continue
            real_deletes.append((store_id, store_path))

        # ── Fast path: nothing to do ─────────────────────────────────────────
        if not candidate_bytes and not real_deletes:
            return warnings

        with telemetry.timed_phase("db_phase"):
            warnings = await _reconcile_db_phase(
                session_id, before, candidate_bytes, real_deletes, warnings
            )
        return warnings


async def _reconcile_db_phase(
    session_id: str,
    before: _Snapshot,
    candidate_bytes: _BytesMap,
    real_deletes: list[tuple[str, str]],
    warnings: list[str],
) -> list[str]:
    """The on-loop DB phase: create/update/delete calls for real work only.

    Split out of :func:`reconcile_memory_mounts` purely so the ``db_phase``
    telemetry span brackets exactly this work (walk/stat/candidate-read is
    already off-loop in :func:`_scan_after`).
    """
    account_id = await sessions_service.load_session_account_id(runtime.require_pool(), session_id)
    pool = runtime.require_pool()
    actor = SessionActor(session_id=session_id)

    # ── New files (created by bash) ─────────────────────────────────────
    for (store_id, store_path), raw in candidate_bytes.items():
        if (store_id, store_path) in before:
            continue  # handled as modify below
        if isinstance(raw, _Unreadable):
            _warn_unreadable(store_id, store_path, warnings, session_id)
            continue
        maybe_content = _read_utf8_content(store_id, store_path, warnings, session_id, raw)
        if maybe_content is None:
            continue
        content = maybe_content
        # create_memory calls _mirror_to_host internally, which rewrites
        # the same bytes bash just wrote. This is redundant I/O but
        # harmless and unavoidable without adding a skip_mirror
        # parameter to the service layer.
        memory = await memory_service.create_memory(
            pool,
            store_id=store_id,
            path=store_path,
            content=content,
            actor=actor,
            account_id=account_id,
        )
        runtime.set_read_sha(session_id, store_id, store_path, memory.content_sha256)
        log.info(
            "memory_reconcile.created",
            session_id=session_id,
            store_id=store_id,
            store_path=store_path,
        )

    # ── Modified candidates (DB-sha anchor) ─────────────────────────────
    # Once the before-pass stops reading bytes, the DB row's
    # content_sha256 is the only available baseline for a modify
    # candidate. Baseline semantics shift honestly here: this diffs
    # DB-state -> disk-after, not disk-before -> disk-after (see module
    # docstring / issue #1748 Behavioral deltas). They coincide for
    # single-session stores and diverge only under shared-store peer-write
    # races, where the DB baseline is the more accurate "state we are
    # updating".
    for (store_id, store_path), raw in candidate_bytes.items():
        if (store_id, store_path) not in before:
            continue  # handled as create above
        if isinstance(raw, _Unreadable):
            _warn_unreadable(store_id, store_path, warnings, session_id)
            continue
        maybe_content = _read_utf8_content(store_id, store_path, warnings, session_id, raw)
        if maybe_content is None:
            continue
        content = maybe_content
        existing = await memory_service.get_memory_by_path(
            pool, store_id, store_path, include_content=False, account_id=account_id
        )
        if existing is None:
            # Race: was in before snapshot but no DB record (e.g.
            # previously skipped binary). Treat as a new create.
            memory = await memory_service.create_memory(
                pool,
                store_id=store_id,
                path=store_path,
                content=content,
                actor=actor,
                account_id=account_id,
            )
            runtime.set_read_sha(session_id, store_id, store_path, memory.content_sha256)
        else:
            candidate_sha = _content_sha256(raw)
            if candidate_sha == existing.content_sha256:
                # Content is unchanged vs DB (a peer session's write beat
                # us to the same content, or a hot/sig-changed candidate
                # turned out byte-identical). No DB write; stamp the read
                # cache so a subsequent tool write in this session isn't
                # spuriously treated as unread.
                runtime.set_read_sha(session_id, store_id, store_path, candidate_sha)
                continue
            # update_memory calls _mirror_to_host internally, which
            # rewrites the same bytes bash just wrote. This is redundant
            # I/O but harmless and unavoidable without adding a
            # skip_mirror parameter to the service layer.
            # Passing existing.content_sha256 as precondition guards
            # against concurrent writes from another session: if a peer
            # session wrote between our get_memory_by_path and this
            # update, the precondition fails and
            # MemoryPreconditionFailedError propagates as a tool-result
            # error, which the model can handle.
            memory = await memory_service.update_memory(
                pool,
                store_id=store_id,
                memory_id=existing.id,
                new_content=content,
                precondition_sha256=existing.content_sha256,
                actor=actor,
                account_id=account_id,
            )
            runtime.set_read_sha(session_id, store_id, store_path, memory.content_sha256)
        log.info(
            "memory_reconcile.modified",
            session_id=session_id,
            store_id=store_id,
            store_path=store_path,
        )

    # ── Deleted files ────────────────────────────────────────────────────
    for store_id, store_path in real_deletes:
        existing = await memory_service.get_memory_by_path(
            pool, store_id, store_path, include_content=False, account_id=account_id
        )
        if existing is None:
            continue  # already gone from DB; nothing to do
        # delete_memory calls _mirror_delete_from_host internally, which
        # tries to unlink a file bash already deleted. Since
        # missing_ok=True it is harmless.
        await memory_service.delete_memory(
            pool,
            store_id=store_id,
            memory_id=existing.id,
            actor=actor,
            account_id=account_id,
        )
        log.info(
            "memory_reconcile.deleted",
            session_id=session_id,
            store_id=store_id,
            store_path=store_path,
        )

    return warnings


# ── Coarse/absent-ctime runtime guard (#1748 step 6) ─────────────────────────


@dataclass(frozen=True, slots=True)
class PrefilterState:
    """Result of the mount-ctime-granularity probe, cached at worker boot."""

    enabled: bool
    observed_granule_ns: int | None


_prefilter_state: PrefilterState | None = None


def probe_mount_ctime_granularity(probe_dir: Path) -> PrefilterState:
    """Probe the actual memory-store mount for ctime support/granularity.

    Writes a temp file twice in quick succession on ``probe_dir`` and
    measures the smallest observed non-zero ctime delta. If ``st_ctime_ns``
    reads back as 0 (no ctime) or the smallest observed delta between two
    back-to-back writes is >= ``HOT_WINDOW_NS`` (a coarser-than-window
    granule), the prefilter is disabled and every candidate is force-hashed —
    the fail-safe direction on the memory-loss axis.

    The common case (stock kernels, coarse-but-sub-window ctime, ~ms tick)
    needs no force-hash: the hot-window already covers it (see module
    docstring). This probe only fires for the exotic case of a granule at or
    above the window itself, or a filesystem with no ctime at all.
    """
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / ".ctime_probe"
    try:
        probe_path.write_bytes(b"a")
        first = os.lstat(probe_path).st_ctime_ns
        probe_path.write_bytes(b"b")
        second = os.lstat(probe_path).st_ctime_ns
    finally:
        probe_path.unlink(missing_ok=True)

    if first == 0 or second == 0:
        return PrefilterState(enabled=False, observed_granule_ns=None)

    delta = abs(second - first)
    if delta >= HOT_WINDOW_NS:
        return PrefilterState(enabled=False, observed_granule_ns=delta)
    return PrefilterState(enabled=True, observed_granule_ns=delta)


def set_prefilter_state(state: PrefilterState) -> None:
    """Install the cached probe result (called once at worker boot)."""
    global _prefilter_state
    _prefilter_state = state


def get_prefilter_state() -> PrefilterState:
    """Return the cached probe result, defaulting to enabled if never probed.

    Never-probed (e.g. unit tests, or a worker that hasn't run boot-time
    probing yet) defaults to the sound-and-common case: prefilter enabled.
    Production boot MUST call :func:`set_prefilter_state` with a real probe
    result before serving bash calls with real memory mounts, per #1748 §6.
    """
    if _prefilter_state is None:
        return PrefilterState(enabled=True, observed_granule_ns=None)
    return _prefilter_state
