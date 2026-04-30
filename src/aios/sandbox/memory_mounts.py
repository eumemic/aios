"""Materialize memory-store content into per-session host directories.

The sandbox bind-mount source for a memory store is a host directory at
``<workspace_root>/<session_id>/memory/<store_name>/``. At container
provisioning time, we read every non-deleted memory in the store from the DB
and write it to that directory so the agent's read tools (and bash reads) see
a consistent snapshot.

Tool-driven writes through write/edit are mirrored to this same directory
after the durable DB write commits — see :mod:`aios.tools.memory_intercept`.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.logging import get_logger
from aios.models.memory_stores import Memory
from aios.sandbox.volumes import memory_dir_for

log = get_logger("aios.sandbox.memory_mounts")


async def materialize_store_to_host(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    store_id: str,
    store_name: str,
) -> None:
    """Populate the host-side mount source for ``store_id`` from the DB.

    Idempotent: deletes the host directory if it exists, then re-materializes.
    Safe to call repeatedly — used at every container wake. The directory
    contents are session-local; cross-session writes don't propagate here.
    """
    host_dir = memory_dir_for(session_id, store_name)
    if host_dir.exists():
        # Wipe stale state from a previous container's lifetime.
        _rm_dir_contents(host_dir)
    host_dir.mkdir(parents=True, exist_ok=True)

    memories = [m for m in await queries.list_memories(conn, store_id) if isinstance(m, Memory)]
    for memory in memories:
        # Reload with content (list_memories omits content for bandwidth).
        full = await queries.get_memory(conn, store_id, memory.id, include_content=True)
        # Memory paths are guaranteed to start with "/" by the SQL CHECK.
        target = host_dir / full.path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(full.content or "")

    log.debug(
        "memory.materialized",
        session_id=session_id,
        store_id=store_id,
        store_name=store_name,
        count=len(memories),
    )


def _rm_dir_contents(path: Any) -> None:
    """Recursively remove a directory's contents but keep the dir itself."""
    import shutil

    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
