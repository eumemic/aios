"""Per-session workspace directory conventions.

Each session gets a stable host-side directory at
``settings.workspace_root / session_id``, which is bind-mounted into the
session's container at ``/workspace``. The directory is created on the
first tool call that provisions a container for the session — chat-only
sessions never create one — and persists across container lifetimes
(container death, session re-wake, worker restart).

The directory is NOT deleted when the container goes away. A session can
be resumed tomorrow and still find its files. Cleanup of stale workspace
dirs is a Phase 6 polish item.
"""

from __future__ import annotations

from pathlib import Path

from aios.config import get_settings


def workspace_dir_for(session_id: str) -> Path:
    """Return the absolute host directory for ``session_id``'s workspace.

    The returned path is always absolute — Docker bind mounts reject
    relative paths. If ``workspace_root`` was configured as a relative
    path (e.g. ``./workspaces`` in a dev ``.env``), it is resolved
    against the current working directory at call time.

    Pure — does not touch the filesystem. Use :func:`ensure_workspace_dir`
    to both compute and create.
    """
    return (get_settings().workspace_root / session_id).resolve()


def ensure_workspace_dir(session_id: str) -> Path:
    """Return the absolute host directory for ``session_id``, creating it if needed.

    Also ensures the parent ``workspace_root`` exists. ``parents=True,
    exist_ok=True`` semantics — safe to call repeatedly.
    """
    path = workspace_dir_for(session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_workspace_path(raw_path: str) -> Path:
    """Resolve ``raw_path`` to an absolute ``Path``, creating it if needed."""
    path = Path(raw_path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


_MEMORY_STORES_ROOT = "_memory_stores"


def memory_stores_root() -> Path:
    """Return ``<workspace_root>/_memory_stores`` — the parent of all
    shared memory-store host directories.

    Per-store host dirs (one per ``memory_store.id``) live as siblings in
    here and are bind-mounted into every attached session's container at
    ``/mnt/memory/<store_name>/``. Sharing the source dir across attached
    sessions is what makes cross-session reads live: a tool write from
    session A appears in session B's mount immediately.
    """
    return (get_settings().workspace_root / _MEMORY_STORES_ROOT).resolve()


def memory_store_host_dir(store_id: str) -> Path:
    """Return the shared host-side directory backing memory store ``store_id``.

    Pure — does not create the directory. Materialization is handled by
    :mod:`aios.sandbox.memory_mounts`, which acquires the matching lock
    file (see :func:`memory_store_lock_path`) before populating from DB.
    """
    return memory_stores_root() / store_id


def memory_store_lock_path(store_id: str) -> Path:
    """Return the file-lock path used to serialize first-attach materialization
    of ``store_id``.

    Two sessions provisioning concurrently for the same store both call
    :func:`materialize_store_to_host`; the lock ensures only one of them
    writes the initial DB snapshot to the host dir. The loser observes
    the ``.materialized`` marker and skips."""
    return memory_stores_root() / f"{store_id}.lock"
