"""Uncorrelated out-of-band full-hash memory audit (#1748 §Uncorrelated detector).

The bash memory-mount reconcile's stat-prefilter (:mod:`aios.tools.bash_memory_reconcile`)
trades content hashing for a cheap ``os.stat`` signature comparison. Its soundness
rests on kernel ctime semantics and filesystem properties that ops can change
underfoot (the memory-store mount is slated to potentially move off the shared
postgres volume — see the runtime probe in that module). A verdict read from
the SAME substrate that produced it (i.e. another stat-based check) is
unfalsifiable from the inside — "substrate-different-verdict".

This module is the falsifying, uncorrelated arm: a low-rate, full-content-hash
sweep that walks every writable memory-store's host directory, hashes every
file's ON-DISK bytes unconditionally (no stat prefilter, no candidate
skipping — the whole point is to NOT reuse the prefilter's code path), and
compares against the DB's ``content_sha256``. Any divergence is alarmed
loudly: this is the external, independent detector that would have caught the
2026-07-04 ultron-memory-rollback class by the machine instead of the
chairman.

Deliberately NOT wired into ``bash_memory_reconcile`` in any way — no shared
helper functions, no shared walk routine beyond the unavoidable
``rglob``/``read_bytes``/``sha256`` primitives every file-hashing sweep must
use. If the prefilter walk logic ever silently regresses (e.g. someone
"optimizes" the symlink-rejection check), this audit's independent walk still
catches divergence because it never trusted the prefilter's candidate set in
the first place.

Cadence: intended to run as a low-rate periodic worker job (mirroring the
other reapers in this package) or an ops-agent cron. Low rate is deliberate —
a full content hash of every memory-store file is exactly the on-loop-hash
cost #1733 forbids doing PER BASH CALL; doing it occasionally, off the hot
path, as a correctness backstop is the intended shape.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg

from aios.db import queries
from aios.logging import get_logger
from aios.sandbox.memory_mounts import MATERIALIZED_MARKER
from aios.sandbox.volumes import memory_store_host_dir, memory_stores_root

log = get_logger("aios.harness.memory_reconcile_audit")


@dataclass(frozen=True, slots=True)
class Divergence:
    """One on-disk-vs-DB content mismatch found by the audit."""

    store_id: str
    store_path: str
    reason: str  # "content_mismatch" | "missing_on_disk" | "missing_in_db" | "unreadable"
    disk_sha256: str | None = None
    db_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class AuditResult:
    stores_checked: int
    files_hashed: int
    divergences: list[Divergence] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.divergences


def _hash_file_unconditionally(fpath: Path) -> str | None:
    """Read and sha256-hash ``fpath``'s full bytes. Returns ``None`` on read failure.

    Deliberately independent of :mod:`aios.tools.bash_memory_reconcile` — no
    stat comparison, no candidate classification, no shared sentinel types.
    Every file this audit walks gets its bytes read and hashed, unconditionally.
    """
    try:
        raw = fpath.read_bytes()
    except OSError:
        return None
    try:
        content = raw.decode("utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except UnicodeDecodeError:
        return hashlib.sha256(raw).hexdigest()


def _walk_all_files(host_dir: Path) -> list[tuple[Path, str]]:
    """Enumerate every real file under ``host_dir`` (independent walk).

    Skips only structural non-content entries (the ``.materialized`` marker,
    in-flight ``.tmp.`` write scratch, symlinks — the same confused-deputy
    threat the reconcile walk guards against, since this audit also runs
    host-side against the same shared directory). This is intentionally the
    ONLY overlap with the reconcile module's walk semantics: both must reject
    symlinks for the same security reason, but neither imports the other's
    walk function.
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


async def _list_all_writable_store_ids(
    conn: asyncpg.Connection[Any],
) -> list[tuple[str, str]]:
    """Return ``(store_id, account_id)`` for every non-archived memory store.

    Cross-account (unscoped) by necessity — this is a worker-level backstop
    audit, not a tenant-facing read; SQL enforces nothing account-specific
    here because the whole point is to check every store this worker's host
    might have materialized, regardless of tenant.
    """
    rows = await conn.fetch(
        "SELECT id, account_id FROM memory_stores WHERE archived_at IS NULL",
    )
    return [(row["id"], row["account_id"]) for row in rows]


async def run_memory_reconcile_audit(pool: asyncpg.Pool[Any]) -> AuditResult:
    """Full-hash disk-vs-DB audit across every materialized, writable memory store.

    For each non-archived memory store whose host directory has been
    materialized on THIS worker: walk every file, hash its on-disk content
    unconditionally, and compare against the DB's live ``content_sha256`` for
    that path. Divergences (content mismatch, present-on-disk-absent-in-DB,
    present-in-DB-absent-on-disk, or unreadable) are collected and logged
    loudly — this function does not raise on divergence (an audit finding is
    not a crash), but every divergence is a structured warning log line the
    ops-agent cron / alerting surface can watch.
    """
    divergences: list[Divergence] = []
    stores_checked = 0
    files_hashed = 0

    async with pool.acquire() as conn:
        store_ids = await _list_all_writable_store_ids(conn)

        for store_id, account_id in store_ids:
            host_dir = memory_store_host_dir(store_id)
            if not host_dir.exists():
                continue  # never materialized on this worker — nothing to audit here
            marker = host_dir / MATERIALIZED_MARKER
            if not marker.exists():
                continue
            stores_checked += 1

            disk_paths = _walk_all_files(host_dir)
            disk_shas: dict[str, str | None] = {}
            for fpath, store_path in disk_paths:
                disk_shas[store_path] = _hash_file_unconditionally(fpath)
                files_hashed += 1

            db_rows = await queries.list_active_memory_paths_and_content(
                conn, store_id, account_id=account_id
            )
            db_shas: dict[str, str] = {
                path: hashlib.sha256(content.encode("utf-8")).hexdigest()
                for path, content in db_rows
            }

            all_paths = set(disk_shas) | set(db_shas)
            for path in all_paths:
                disk_sha = disk_shas.get(path)
                db_sha = db_shas.get(path)
                if path in disk_shas and path not in db_shas:
                    divergences.append(
                        Divergence(
                            store_id=store_id,
                            store_path=path,
                            reason="missing_in_db",
                            disk_sha256=disk_sha,
                        )
                    )
                elif path not in disk_shas and path in db_shas:
                    divergences.append(
                        Divergence(
                            store_id=store_id,
                            store_path=path,
                            reason="missing_on_disk",
                            db_sha256=db_sha,
                        )
                    )
                elif disk_sha is None:
                    divergences.append(
                        Divergence(
                            store_id=store_id,
                            store_path=path,
                            reason="unreadable",
                            db_sha256=db_sha,
                        )
                    )
                elif disk_sha != db_sha:
                    divergences.append(
                        Divergence(
                            store_id=store_id,
                            store_path=path,
                            reason="content_mismatch",
                            disk_sha256=disk_sha,
                            db_sha256=db_sha,
                        )
                    )

    for d in divergences:
        log.warning(
            "memory_reconcile_audit.divergence",
            store_id=d.store_id,
            store_path=d.store_path,
            reason=d.reason,
            disk_sha256=d.disk_sha256,
            db_sha256=d.db_sha256,
        )

    log.info(
        "memory_reconcile_audit.swept",
        stores_checked=stores_checked,
        files_hashed=files_hashed,
        divergences=len(divergences),
    )

    return AuditResult(
        stores_checked=stores_checked,
        files_hashed=files_hashed,
        divergences=divergences,
    )


def memory_reconcile_audit_probe_root() -> Path:
    """Return the root the audit scans stores under (test seam / documentation)."""
    return memory_stores_root()
