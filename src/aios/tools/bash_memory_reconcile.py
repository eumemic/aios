"""Post-exec memory-mount reconciliation for the bash tool.

After each bash command completes, we diff the memory-mount host directories
against a snapshot taken before the command ran. Any files that were created,
modified, or deleted by the bash command are propagated to the database as
``memory_versions`` rows, closing the v2 limitation where bash writes were
visible cross-session via the shared FS but did not produce durable version
history.

Design constraints:
- Snapshot is synchronous (no DB I/O) — taken before ``sandbox.exec``.
- Reconcile is async — runs after ``sandbox.exec`` returns.
- read_only mounts are never written; we skip them in both directions.
- Binary files and oversized files emit warnings instead of failing loudly
  (the command already ran; we just can't record what it wrote).
- sha256 is over UTF-8 encoded bytes of the content string, matching the
  convention in :mod:`aios.services.memory_stores._sha256_hex`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from aios.harness import runtime
from aios.logging import get_logger
from aios.models.memory_stores import MAX_CONTENT_BYTES
from aios.sandbox.memory_mounts import MATERIALIZED_MARKER
from aios.sandbox.volumes import memory_store_host_dir
from aios.services import memory_stores as memory_service
from aios.services import sessions as sessions_service
from aios.services.memory_stores import SessionActor

log = get_logger("aios.tools.bash_memory_reconcile")

# Snapshot type: (store_id, store_path) -> sha256_hex
_Snapshot = dict[tuple[str, str], str]


def _bytes_map_to_sha_map(
    bytes_map: dict[tuple[str, str], bytes],
) -> _Snapshot:
    """Convert a raw-bytes map to a sha256-hex map.

    For UTF-8 decodable bytes, sha is over the decoded string (matching the
    memory_stores convention).  For binary blobs, sha is over the raw bytes —
    this allows change detection even though binary files cannot be stored.
    """
    result: _Snapshot = {}
    for (store_id, store_path), raw in bytes_map.items():
        try:
            content = raw.decode("utf-8")
            # sha256 over UTF-8 bytes of content — mirrors memory_stores._sha256_hex
            sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        except UnicodeDecodeError:
            sha = hashlib.sha256(raw).hexdigest()
        result[(store_id, store_path)] = sha
    return result


def _walk_store_files(host_dir: Path) -> list[tuple[Path, str]]:
    """Return (absolute_path, store_path) for all eligible files in host_dir.

    Skips directories, the .materialized marker, any hidden temp files
    starting with ``.tmp.`` (written by the atomic_mirror helper during
    in-flight writes), and any symlinks. The symlink rejection mirrors
    PR #497's policy for ``walk_skill_dir``: bash inside the sandbox can
    ``ln -s <worker-side-path> /mnt/memory/<store>/leak``, and the
    subsequent host-side ``read_bytes`` would resolve the symlink
    against the worker's filesystem — exfiltrating any worker-readable
    file (vault key material on disk, neighbour tenants' state, /etc,
    OAuth refresh tokens, …) into a memory store that then persists
    the bytes to DB and renders them to the model on every wake.
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


def _snapshot_with_bytes(
    session_id: str,
) -> dict[tuple[str, str], bytes]:
    """Return raw bytes for all eligible writable materialized mounts.

    Called internally by both ``snapshot_memory_mounts`` (which only needs
    sha digests) and ``reconcile_memory_mounts`` (which needs bytes to avoid
    a second scan).

    Returns ``bytes_map``: ``(store_id, store_path) -> raw_bytes``.
    """
    echoes = runtime.get_session_memory_mounts(session_id)
    bytes_map: dict[tuple[str, str], bytes] = {}

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

        for fpath, store_path in _walk_store_files(host_dir):
            try:
                raw = fpath.read_bytes()
            except OSError:
                continue
            bytes_map[(store_id, store_path)] = raw

    return bytes_map


def snapshot_memory_mounts(session_id: str) -> _Snapshot:
    """Return a sha256 snapshot of all writable, materialized memory mount files.

    Called synchronously before ``sandbox.exec``.  No DB I/O.  Returns a dict
    mapping ``(store_id, store_path)`` to the sha256 hex digest of the file's
    content at that point.

    Skips:
    - Sessions with no attached stores.
    - read_only mounts.
    - Stores whose host directory does not exist yet.
    - Stores that have not been materialized (marker file absent).
    """
    by_bytes = _snapshot_with_bytes(session_id)
    return _bytes_map_to_sha_map(by_bytes)


def _read_utf8_content(
    store_id: str,
    store_path: str,
    warnings: list[str],
    session_id: str,
    raw: bytes,
) -> str | None:
    """Decode ``raw`` as UTF-8 and enforce the size limit.

    Returns the decoded string on success, or ``None`` if the bytes cannot be
    decoded or exceed ``MAX_CONTENT_BYTES``.  In the rejection case a human-
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


async def reconcile_memory_mounts(session_id: str, before: _Snapshot) -> list[str]:
    """Diff memory mount state against ``before``; write DB changes.

    Called after ``sandbox.exec`` returns.  For each writable mount:

    - New files (in after, not in before): ``create_memory``
    - Modified files (sha changed): ``update_memory`` with precondition
    - Deleted files (in before, not in after): ``delete_memory``
    - Unchanged files: no DB call

    Binary files and files exceeding ``MAX_CONTENT_BYTES`` are skipped with a
    warning string (collected and returned).  DB errors propagate — they are
    the session's problem to recover from through the normal error channel.
    """
    account_id = await sessions_service.load_session_account_id(runtime.require_pool(), session_id)
    # Build after snapshot as (store_id, store_path) -> bytes in one pass.
    # Using bytes avoids re-reading files during the create/modify loops.
    after_bytes = _snapshot_with_bytes(session_id)
    pool = runtime.require_pool()
    actor = SessionActor(session_id=session_id)
    warnings: list[str] = []

    # Compute sha for after entries (needed for equality checks).
    after: _Snapshot = _bytes_map_to_sha_map(after_bytes)

    # ── New files (created by bash) ─────────────────────────────────────────
    for (store_id, store_path), _sha in after.items():
        if (store_id, store_path) in before:
            continue  # will be handled as modify or unchanged
        raw = after_bytes[(store_id, store_path)]
        maybe_content = _read_utf8_content(
            store_id,
            store_path,
            warnings,
            session_id,
            raw,
        )
        if maybe_content is None:
            continue
        content = maybe_content
        # create_memory calls _mirror_to_host internally, which rewrites the same
        # bytes bash just wrote. This is redundant I/O but harmless and unavoidable
        # without adding a skip_mirror parameter to the service layer.
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

    # ── Modified files (sha changed) ────────────────────────────────────────
    for (store_id, store_path), before_sha in before.items():
        after_sha = after.get((store_id, store_path))
        if after_sha is None:
            continue  # deleted — handled below
        if after_sha == before_sha:
            continue  # unchanged
        raw = after_bytes[(store_id, store_path)]
        maybe_content = _read_utf8_content(
            store_id,
            store_path,
            warnings,
            session_id,
            raw,
        )
        if maybe_content is None:
            continue
        content = maybe_content
        existing = await memory_service.get_memory_by_path(
            pool, store_id, store_path, include_content=False, account_id=account_id
        )
        if existing is None:
            # Race: was in before snapshot but no DB record (e.g. previously skipped binary).
            # Treat as a new create.
            # create_memory calls _mirror_to_host internally, which rewrites the same
            # bytes bash just wrote. This is redundant I/O but harmless and unavoidable
            # without adding a skip_mirror parameter to the service layer.
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
            # update_memory calls _mirror_to_host internally, which rewrites the same
            # bytes bash just wrote. This is redundant I/O but harmless and unavoidable
            # without adding a skip_mirror parameter to the service layer.
            # Passing existing.content_sha256 as precondition guards against concurrent writes
            # from another session: if a peer session wrote between our get_memory_by_path and
            # this update, the precondition fails and MemoryPreconditionFailedError propagates
            # as a tool-result error, which the model can handle.
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

    # ── Deleted files ────────────────────────────────────────────────────────
    for store_id, store_path in before:
        if (store_id, store_path) in after:
            continue  # still exists
        existing = await memory_service.get_memory_by_path(
            pool, store_id, store_path, include_content=False, account_id=account_id
        )
        if existing is None:
            continue  # already gone from DB; nothing to do
        # delete_memory calls _mirror_delete_from_host internally, which tries to unlink
        # a file bash already deleted. Since missing_ok=True it is harmless.
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
