"""Detect when a file-tool path lives under a session's memory mount.

A v1 design choice (see plan): only ``write`` and ``edit`` need the
intercept — reads pass through to the bind-mounted host directory, which
is materialized at session wake and kept in sync by tool writes.

The intercept exists for two reasons:

1. **Durability**. Bash writes are out-of-band by design, but tool writes
   should produce immutable ``memory_versions`` rows. Without the
   intercept, a tool write would only land on the host bind-mount —
   correct for in-session reads, but lost the moment a fresh session
   re-materializes from DB.

2. **Access enforcement at the model layer**. ``:ro`` mounts are kernel-
   enforced, but a kernel ``EROFS`` is opaque to the model. Returning a
   typed tool error from the intercept gives the agent a meaningful
   message it can act on.

The mount cache lives on :mod:`aios.harness.runtime`, populated at the top
of every step in ``loop._run_session_step_body``.
"""

from __future__ import annotations

from dataclasses import dataclass

from aios.harness import runtime
from aios.models.memory_stores import Access, MemoryStoreResourceEcho


@dataclass(frozen=True)
class MemoryTarget:
    """The memory-store-relative addressing of a file path under a mount."""

    store_id: str
    store_name: str
    mount_path: str
    store_path: str  # path inside the store, always begins with "/"
    access: Access


def resolve_memory_target(session_id: str, fs_path: str) -> MemoryTarget | None:
    """Return target metadata if ``fs_path`` lives under a memory mount.

    ``fs_path`` is the value the model passed as the ``path`` / ``file_path``
    argument. It must be absolute and exactly under ``/mnt/memory/<store>/``
    for the call to count as a memory-mount target — relative paths and
    paths that traverse out via ``..`` are not memory targets and pass
    through to the regular sandbox FS path. (This mirrors Anthropic's
    behavior: the mount source is a regular directory; nothing magical
    happens for paths that don't actually live inside it.)
    """
    if not fs_path.startswith("/mnt/memory/"):
        return None
    echoes: list[MemoryStoreResourceEcho] = runtime.get_session_memory_mounts(session_id)
    for echo in echoes:
        prefix = echo.mount_path.rstrip("/") + "/"
        if fs_path == echo.mount_path or fs_path.startswith(prefix):
            store_path = fs_path[len(echo.mount_path) :] or "/"
            if not store_path.startswith("/"):
                store_path = "/" + store_path
            return MemoryTarget(
                store_id=echo.memory_store_id,
                store_name=echo.name,
                mount_path=echo.mount_path,
                store_path=store_path,
                access=echo.access,
            )
    return None
