"""Worker-startup repair pass for shared-workspace ownership (#959).

The worker container runs as root; the api runs as uid 1000. A shared dir
the worker created first (before the ``ensure_owned_dir`` fix shipped) is
``root:root`` and the api (no CAP_CHOWN) can't write into it. This pass —
called from ``worker_main`` before any provisioning job can run — chowns
pre-existing root-owned (or otherwise mismatched) entries on a *bounded*
frontier of the shared tree back to the configured owner.

Bounded, NOT a deep walk: ``workspace_root`` itself, its first-level
entries, and the immediate children of the ``_``-prefixed shared roots
and per-account/legacy-session dirs. Prod layout is
``workspace_root/{account_id}/{session_id}``, so per-session dirs are
children of ``acc_*`` — covering ``acc_*`` children cleans the legacy
root-owned ``sess_*`` residue the issue calls out without an unbounded
recursion that would touch user content.
"""

from __future__ import annotations

import os
from pathlib import Path

from aios.config import get_settings
from aios.logging import get_logger
from aios.sandbox.volumes import (
    _ATTACHMENTS_ROOT,
    _GITHUB_REPOS_ROOT,
    _MEMORY_STORES_ROOT,
    _SESSION_REPOS_ROOT,
    _UPLOADS_ROOT,
)

log = get_logger("aios.worker")

# The ``_``-prefixed shared roots whose immediate children must also be
# checked. Sourced from volumes.py's dir-name constants so the two stay
# in lockstep.
_SHARED_ROOTS = frozenset(
    {
        _UPLOADS_ROOT,
        _ATTACHMENTS_ROOT,
        _MEMORY_STORES_ROOT,
        _GITHUB_REPOS_ROOT,
        _SESSION_REPOS_ROOT,
    }
)

# Per-account dirs (post-#409 layout) and legacy per-session dirs whose
# immediate children are descended one level. Matched by explicit prefix
# rather than "any non-underscore dir" so unrelated content isn't chowned.
_DESCEND_PREFIXES = ("acc_", "sess_")


def repair_workspace_ownership() -> int:
    """Repair root-owned entries on the shared workspace tree to the configured
    owner uid:gid. Bounded (NOT a deep walk). Returns count repaired. No-op
    (returns 0) when euid != 0."""
    if os.geteuid() != 0:
        log.debug("workspace.ownership_repair_skipped", reason="not_root")
        return 0

    settings = get_settings()
    root = settings.workspace_root.resolve()
    uid, gid = settings.workspaces_owner_uid, settings.workspaces_owner_gid
    if not root.exists():
        return 0

    def _repair_one(path: Path) -> bool:
        try:
            st = path.lstat()  # lstat: chown the symlink itself, not its target
            if st.st_uid != uid or st.st_gid != gid:
                os.chown(path, uid, gid)
                log.info(
                    "workspace.ownership_repaired",
                    path=str(path),
                    old_uid=st.st_uid,
                    old_gid=st.st_gid,
                )
                return True
            return False
        except OSError as e:
            # One bad entry must not crash worker startup.
            log.info("workspace.ownership_repair_failed", path=str(path), error=str(e))
            return False

    count = 0
    if _repair_one(root):
        count += 1
    for entry in root.iterdir():
        if _repair_one(entry):
            count += 1
        # Descend exactly one level into shared roots and account/legacy dirs.
        if not entry.is_dir():
            continue
        if entry.name in _SHARED_ROOTS or entry.name.startswith(_DESCEND_PREFIXES):
            for child in entry.iterdir():
                if _repair_one(child):
                    count += 1
    return count
