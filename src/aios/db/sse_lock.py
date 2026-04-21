"""Advisory-lock-based SSE subscriber detection (issue #81).

The dynamic-streaming decision — "is anyone watching this session's SSE
stream right now?" — is answered by whether a backend holds a shared
advisory lock keyed to the session.  When an SSE endpoint opens, it
grabs the lock on its dedicated connection.  When the connection dies
(client disconnect, API crash, TCP reset), Postgres releases the lock
automatically: no tracking table, no cleanup job.

The worker queries ``pg_locks`` before each step to decide whether the
model call should stream (deltas → ``pg_notify`` → SSE clients) or run
non-streaming (faster end-to-end on proxies like OpenRouter→Groq when
nobody is consuming deltas).
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg


def session_lock_key(session_id: str) -> tuple[int, int]:
    """Derive a ``(classid, objid)`` advisory-lock key pair from ``session_id``.

    BLAKE2b hashes the session identifier into 8 bytes, then splits into
    two signed int32s so the pair maps directly onto ``pg_locks.classid``
    and ``pg_locks.objid``.  Deterministic; collision probability is
    negligible for realistic deployments.
    """
    digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=8).digest()
    classid, objid = struct.unpack("<ii", digest)
    return classid, objid


async def acquire_subscriber_lock(conn: asyncpg.Connection[Any], session_id: str) -> None:
    """Hold a shared advisory lock on ``conn`` for the duration of the connection.

    Multiple subscribers coexist via shared-mode.  The lock releases
    automatically when ``conn`` closes (client disconnect, crash, TCP
    reset) — that auto-release is the whole point of the pattern.

    The ``::int`` casts are load-bearing: asyncpg infers the parameter
    type as ``oid`` (uint32) from context if we don't pin it, which
    fails for negative keys coming out of the BLAKE2 hash.
    """
    classid, objid = session_lock_key(session_id)
    await conn.execute("SELECT pg_advisory_lock_shared($1::int, $2::int)", classid, objid)


async def has_subscriber(pool: asyncpg.Pool[Any], session_id: str) -> bool:
    """Return True if any backend holds a shared subscriber lock for this session.

    ``pg_locks.classid`` / ``objid`` are ``oid`` (uint32) but our keys are
    signed int32.  Casting the parameter as ``int::oid`` reinterprets the
    sign bit so equality works correctly across the signed/unsigned
    boundary.  The ``database = (...)`` predicate scopes the scan to the
    current database so a busy cluster's cross-db locks don't bloat this
    hot-path query.
    """
    classid, objid = session_lock_key(session_id)
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_locks "
            "WHERE locktype = 'advisory' "
            "AND database = (SELECT oid FROM pg_database WHERE datname = current_database()) "
            "AND classid = ($1::int)::oid "
            "AND objid = ($2::int)::oid "
            "AND mode = 'ShareLock')",
            classid,
            objid,
        )
    return bool(result)
