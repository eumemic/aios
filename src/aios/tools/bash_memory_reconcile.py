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

import structlog

from aios.harness import runtime
from aios.models.memory_stores import MAX_CONTENT_BYTES
from aios.sandbox.memory_mounts import _MATERIALIZED_MARKER
from aios.sandbox.volumes import memory_store_host_dir
from aios.services import memory_stores as memory_service
from aios.services.memory_stores import SessionActor

log = structlog.get_logger("aios.tools.bash_memory_reconcile")

# Snapshot type: (store_id, store_path) -> sha256_hex
_Snapshot = dict[tuple[str, str], str]


def _sha256_of_content(content: str) -> str:
    """sha256 over UTF-8 bytes of content — matches memory_stores._sha256_hex."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _walk_store_files(host_dir: Path) -> list[tuple[Path, str]]:
    """Return (absolute_path, store_path) for all eligible files in host_dir.

    Skips directories, the .materialized marker, and any hidden temp files
    starting with ``.tmp.`` (written by the atomic_mirror helper during
    in-flight writes).
    """
    results: list[tuple[Path, str]] = []
    for fpath in host_dir.rglob("*"):
        if fpath.is_dir():
            continue
        name = fpath.name
        if name == _MATERIALIZED_MARKER:
            continue
        if name.startswith(".tmp."):
            continue
        rel = fpath.relative_to(host_dir)
        store_path = "/" + str(rel).replace("\\", "/")
        results.append((fpath, store_path))
    return results


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
    echoes = runtime.get_session_memory_mounts(session_id)
    result: _Snapshot = {}

    for echo in echoes:
        if echo.access == "read_only":
            continue
        store_id = echo.memory_store_id
        host_dir = memory_store_host_dir(store_id)
        if not host_dir.exists():
            continue
        marker = host_dir / _MATERIALIZED_MARKER
        if not marker.exists():
            continue

        for fpath, store_path in _walk_store_files(host_dir):
            try:
                raw = fpath.read_bytes()
            except OSError:
                continue
            # sha256 over UTF-8 decoded content bytes to stay consistent with
            # _sha256_hex in memory_stores service (which hashes content.encode("utf-8"))
            try:
                content = raw.decode("utf-8")
                sha = _sha256_of_content(content)
            except UnicodeDecodeError:
                # Binary file — still snapshot with raw-bytes sha so we can detect
                # changes, but reconcile will skip it with a warning.
                sha = hashlib.sha256(raw).hexdigest()
            result[(store_id, store_path)] = sha

    return result


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
    after = snapshot_memory_mounts(session_id)
    pool = runtime.require_pool()
    actor = SessionActor(session_id=session_id)
    warnings: list[str] = []

    # Build store_id -> host_dir mapping for quick lookup during reconcile.
    echoes = runtime.get_session_memory_mounts(session_id)
    host_dirs: dict[str, Path] = {}
    for echo in echoes:
        if echo.access == "read_only":
            continue
        hd = memory_store_host_dir(echo.memory_store_id)
        if hd.exists() and (hd / _MATERIALIZED_MARKER).exists():
            host_dirs[echo.memory_store_id] = hd

    # ── New files (created by bash) ─────────────────────────────────────────
    for (store_id, store_path), _sha in after.items():
        if (store_id, store_path) in before:
            continue  # will be handled as modify or unchanged
        host_dir = host_dirs.get(store_id)
        if host_dir is None:
            continue
        fpath = host_dir / store_path.lstrip("/")
        try:
            raw = fpath.read_bytes()
        except OSError:
            continue
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
            continue
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
            continue
        memory = await memory_service.create_memory(
            pool,
            store_id=store_id,
            path=store_path,
            content=content,
            actor=actor,
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
        host_dir = host_dirs.get(store_id)
        if host_dir is None:
            continue
        fpath = host_dir / store_path.lstrip("/")
        try:
            raw = fpath.read_bytes()
        except OSError:
            continue
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
            continue
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
            continue
        existing = await memory_service.get_memory_by_path(
            pool, store_id, store_path, include_content=False
        )
        if existing is None:
            # Race: was in before snapshot but no DB record (e.g. previously skipped binary).
            # Treat as a new create.
            memory = await memory_service.create_memory(
                pool,
                store_id=store_id,
                path=store_path,
                content=content,
                actor=actor,
            )
            runtime.set_read_sha(session_id, store_id, store_path, memory.content_sha256)
        else:
            memory = await memory_service.update_memory(
                pool,
                store_id=store_id,
                memory_id=existing.id,
                new_content=content,
                precondition_sha256=before_sha,
                actor=actor,
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
        if store_id not in host_dirs:
            continue
        existing = await memory_service.get_memory_by_path(
            pool, store_id, store_path, include_content=False
        )
        if existing is None:
            continue  # already gone from DB; nothing to do
        await memory_service.delete_memory(
            pool,
            store_id=store_id,
            memory_id=existing.id,
            actor=actor,
        )
        log.info(
            "memory_reconcile.deleted",
            session_id=session_id,
            store_id=store_id,
            store_path=store_path,
        )

    return warnings
