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
