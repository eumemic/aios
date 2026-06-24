"""Memory-store queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.crypto.vault import EncryptedBlob
from aios.db.queries import (
    _archive_scoped,
    _build_set_assignments,
    _escape_like,
    parse_jsonb,
)
from aios.errors import (
    ConflictError,
    MemoryPathConflictError,
    MemoryPreconditionFailedError,
    MemoryStoreArchivedError,
    NotFoundError,
)
from aios.ids import (
    GITHUB_REPOSITORY,
    MEMORY,
    MEMORY_STORE,
    MEMORY_VERSION,
    make_id,
)
from aios.models.github_repositories import GithubRepositoryResourceEcho
from aios.models.memory_stores import (
    Access,
    Actor,
    ActorType,
    Memory,
    MemoryPrefix,
    MemoryStore,
    MemoryStoreResource,
    MemoryStoreResourceEcho,
    MemoryVersion,
)

# ─── memory stores ──────────────────────────────────────────────────────────


def _row_to_memory_store(row: asyncpg.Record) -> MemoryStore:
    raw_metadata = row["metadata"]
    metadata = parse_jsonb(raw_metadata)
    return MemoryStore(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _row_to_memory(row: asyncpg.Record, *, include_content: bool) -> Memory:
    return Memory(
        id=row["id"],
        memory_store_id=row["memory_store_id"],
        memory_version_id=row["current_version_id"],
        path=row["path"],
        content=row["content"] if include_content else None,
        content_sha256=row["content_sha256"],
        content_size_bytes=row["content_size_bytes"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_memory_version(row: asyncpg.Record, *, include_content: bool) -> MemoryVersion:
    redacted = row["redacted_at"] is not None
    redacted_by: Actor | None = None
    if redacted and row["redacted_by_type"] is not None:
        redacted_by = _build_actor(row["redacted_by_type"], row["redacted_by_ref"])
    return MemoryVersion(
        id=row["id"],
        memory_store_id=row["memory_store_id"],
        memory_id=row["memory_id"],
        operation=row["operation"],
        path=row["path"],
        content=row["content"] if include_content and not redacted else None,
        content_sha256=row["content_sha256"],
        content_size_bytes=row["content_size_bytes"],
        created_by=_build_actor(row["created_by_type"], row["created_by_ref"]),
        created_at=row["created_at"],
        redacted_at=row["redacted_at"],
        redacted_by=redacted_by,
    )


def _build_actor(actor_type: str, actor_ref: str) -> Actor:
    if actor_type == "session_actor":
        return Actor(type="session_actor", session_id=actor_ref)
    return Actor(type="api_actor", api_key_id=actor_ref)


# Stores ───────────────────────────────────────────────────────────────────


async def insert_memory_store(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    name: str,
    description: str,
    metadata: dict[str, Any],
) -> MemoryStore:
    row = await conn.fetchrow(
        """
        INSERT INTO memory_stores (id, name, description, metadata, account_id)
        VALUES ($1, $2, $3, $4::jsonb, $5)
        RETURNING *
        """,
        make_id(MEMORY_STORE),
        name,
        description,
        json.dumps(metadata),
        account_id,
    )
    assert row is not None
    return _row_to_memory_store(row)


async def get_memory_store(
    conn: asyncpg.Connection[Any], store_id: str, *, account_id: str, allow_archived: bool = True
) -> MemoryStore:
    row = await conn.fetchrow(
        "SELECT * FROM memory_stores WHERE id = $1 AND account_id = $2",
        store_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(f"memory store {store_id} not found", detail={"id": store_id})
    store = _row_to_memory_store(row)
    if not allow_archived and store.archived_at is not None:
        raise MemoryStoreArchivedError(
            f"memory store {store_id} is archived",
            detail={"id": store_id},
        )
    return store


async def list_memory_stores(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    include_archived: bool = False,
    limit: int = 100,
    after: str | None = None,
) -> list[MemoryStore]:
    args: list[Any] = [account_id]
    where = ["account_id = $1"]
    if not include_archived:
        where.append("archived_at IS NULL")
    if after is not None:
        args.append(after)
        where.append(f"id < ${len(args)}")
    args.append(limit)
    sql = (
        f"SELECT * FROM memory_stores WHERE {' AND '.join(where)} "
        f"ORDER BY id DESC LIMIT ${len(args)}"
    )
    rows = await conn.fetch(sql, *args)
    return [_row_to_memory_store(r) for r in rows]


async def update_memory_store(
    conn: asyncpg.Connection[Any],
    store_id: str,
    *,
    account_id: str,
    name: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MemoryStore:
    # Refuse updates to archived stores: the read path filters
    # ``archived_at IS NULL``, so a rewrite of an archived row has
    # no observable effect — but the bare UPDATE below would still
    # commit the new values and the RETURNING-built response would
    # lie back to the caller as if the update took.  Same shape as
    # ``update_agent`` / ``update_environment`` / ``update_session``
    # (PR #573) / ``update_session_template`` (PR #547) /
    # ``update_vault`` (PR #554).  Defense-in-depth for callers
    # that bypass the service layer (services/memory_stores.py
    # already pre-checks via the equivalent ``allow_archived=False``
    # shape).
    current = await get_memory_store(conn, store_id, allow_archived=False, account_id=account_id)

    args: list[Any] = [store_id]
    fields: list[tuple[str, Any, str | None]] = []
    if name is not None:
        fields.append(("name", name, None))
    if description is not None:
        fields.append(("description", description, None))
    if metadata is not None:
        fields.append(("metadata", metadata, "jsonb"))
    sets = _build_set_assignments(fields, args)
    if not sets:
        return current
    sets.append("updated_at = now()")
    args.append(account_id)
    sql = (
        f"UPDATE memory_stores SET {', '.join(sets)} "
        f"WHERE id = $1 AND account_id = ${len(args)} AND archived_at IS NULL RETURNING *"
    )
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise ConflictError(f"memory store {store_id} is archived", detail={"id": store_id})
    return _row_to_memory_store(row)


async def archive_memory_store(
    conn: asyncpg.Connection[Any], store_id: str, *, account_id: str, idempotent: bool = False
) -> MemoryStore:
    row = await _archive_scoped(
        conn,
        table="memory_stores",
        id_=store_id,
        account_id=account_id,
        noun="memory store",
        idempotent=idempotent,
    )
    return _row_to_memory_store(row)


async def delete_memory_store(
    conn: asyncpg.Connection[Any], store_id: str, *, account_id: str
) -> None:
    result = await conn.execute(
        "DELETE FROM memory_stores WHERE id = $1 AND account_id = $2",
        store_id,
        account_id,
    )
    if result == "DELETE 0":
        raise NotFoundError(f"memory store {store_id} not found", detail={"id": store_id})


# Memory + version (single-txn helpers) ────────────────────────────────────


async def _allocate_version_seq(
    conn: asyncpg.Connection[Any], store_id: str, *, account_id: str
) -> int:
    """Bump ``last_version_seq`` on the store row and return the allocated seq.

    Mirror of the events seq allocation at append_event: row-lock the parent,
    increment, return. Caller must be inside a transaction so the seq is
    bound to the version insert that follows.
    """
    row = await conn.fetchrow(
        "UPDATE memory_stores SET last_version_seq = last_version_seq + 1, "
        "updated_at = now() WHERE id = $1 AND archived_at IS NULL AND account_id = $2 "
        "RETURNING last_version_seq",
        store_id,
        account_id,
    )
    if row is None:
        existing = await conn.fetchrow(
            "SELECT archived_at FROM memory_stores WHERE id = $1 AND account_id = $2",
            store_id,
            account_id,
        )
        if existing is None:
            raise NotFoundError(f"memory store {store_id} not found", detail={"id": store_id})
        raise MemoryStoreArchivedError(
            f"memory store {store_id} is archived",
            detail={"id": store_id},
        )
    seq: int = row["last_version_seq"]
    return seq


async def insert_memory_with_version(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    store_id: str,
    path: str,
    content: str,
    content_sha256: str,
    actor_type: ActorType,
    actor_ref: str,
) -> Memory:
    """Insert a new memory + its initial ``created`` version in one txn.

    On path collision raises :class:`MemoryPathConflictError` carrying the
    existing memory id. The caller can decide between updating that memory
    and surfacing the error.
    """
    size_bytes = len(content.encode("utf-8"))
    memory_id = make_id(MEMORY)
    version_id = make_id(MEMORY_VERSION)

    try:
        async with conn.transaction():
            seq = await _allocate_version_seq(conn, store_id, account_id=account_id)

            # Version first — its `memory_id` column is non-FK, so the
            # not-yet-inserted memory row doesn't block this. Memory row
            # references back via current_version_id.
            await conn.execute(
                """
                INSERT INTO memory_versions
                    (id, memory_store_id, memory_id, seq, operation, path,
                     content, content_sha256, content_size_bytes,
                     created_by_type, created_by_ref, account_id)
                VALUES ($1, $2, $3, $4, 'created', $5, $6, $7, $8, $9, $10, $11)
                """,
                version_id,
                store_id,
                memory_id,
                seq,
                path,
                content,
                content_sha256,
                size_bytes,
                actor_type,
                actor_ref,
                account_id,
            )

            row = await conn.fetchrow(
                """
                INSERT INTO memories
                    (id, memory_store_id, path, content, content_sha256,
                     content_size_bytes, current_version_id, account_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING *
                """,
                memory_id,
                store_id,
                path,
                content,
                content_sha256,
                size_bytes,
                version_id,
                account_id,
            )
    except asyncpg.UniqueViolationError as exc:
        # Re-issue the lookup outside the rolled-back transaction so the
        # error envelope can carry the existing memory id.
        existing = await conn.fetchrow(
            "SELECT id FROM memories WHERE memory_store_id = $1 AND path = $2 "
            "AND deleted_at IS NULL",
            store_id,
            path,
        )
        conflicting_id = existing["id"] if existing is not None else None
        raise MemoryPathConflictError(
            f"path {path!r} is already used by {conflicting_id!r}; use update to modify it",
            detail={
                "conflicting_memory_id": conflicting_id,
                "conflicting_path": path,
            },
        ) from exc
    assert row is not None
    return _row_to_memory(row, include_content=False)


async def get_memory(
    conn: asyncpg.Connection[Any],
    store_id: str,
    memory_id: str,
    *,
    account_id: str,
    include_content: bool = True,
) -> Memory:
    row = await conn.fetchrow(
        "SELECT * FROM memories WHERE memory_store_id = $1 AND id = $2 "
        "AND deleted_at IS NULL AND account_id = $3",
        store_id,
        memory_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"memory {memory_id} not found in store {store_id}",
            detail={"id": memory_id, "memory_store_id": store_id},
        )
    return _row_to_memory(row, include_content=include_content)


async def get_memory_by_path(
    conn: asyncpg.Connection[Any],
    store_id: str,
    path: str,
    *,
    account_id: str,
    include_content: bool = True,
) -> Memory | None:
    row = await conn.fetchrow(
        "SELECT * FROM memories WHERE memory_store_id = $1 AND path = $2 "
        "AND deleted_at IS NULL AND account_id = $3",
        store_id,
        path,
        account_id,
    )
    if row is None:
        return None
    return _row_to_memory(row, include_content=include_content)


async def list_active_memory_paths_and_content(
    conn: asyncpg.Connection[Any],
    store_id: str,
    *,
    account_id: str,
) -> list[tuple[str, str]]:
    """Bulk-fetch ``(path, content)`` for every non-deleted memory in the store.

    Used by sandbox materialization, which needs all live memories in one
    DB roundtrip rather than ``list_memories`` (metadata only) followed by
    a per-memory ``get_memory(include_content=True)`` fan-out.

    ``account_id`` is enforced in SQL even though the upstream caller has
    already account-validated ``store_id`` via
    ``list_session_memory_store_echoes`` — defense in depth so the
    materializer can't be coerced into reading another tenant's memories.
    """
    rows = await conn.fetch(
        "SELECT path, content FROM memories "
        "WHERE memory_store_id = $1 AND account_id = $2 AND deleted_at IS NULL",
        store_id,
        account_id,
    )
    return [(r["path"], r["content"]) for r in rows]


async def list_memories(
    conn: asyncpg.Connection[Any],
    store_id: str,
    *,
    account_id: str,
    path_prefix: str | None = None,
    order_by: str = "created_at",
    depth: int | None = None,
    limit: int = 100,
) -> list[Memory | MemoryPrefix]:
    """List memories, optionally filtered by ``path_prefix`` and depth-clipped.

    ``depth`` requires ``order_by='path'`` (matches Anthropic's wire validation).
    With depth set, paths whose component count under the prefix exceeds
    ``depth`` are collapsed into ``memory_prefix`` synthetic entries. The
    ``limit`` caps the raw-row fetch — depth aggregation may then collapse
    that into fewer response entries, but the SQL bound prevents unbounded
    payloads on stores with thousands of memories.
    """
    if depth is not None and order_by != "path":
        raise ConflictError(
            "depth requires order_by=path",
            detail={"order_by": order_by, "depth": depth},
        )

    where = "memory_store_id = $1 AND deleted_at IS NULL AND account_id = $2"
    args: list[Any] = [store_id, account_id]
    if path_prefix:
        args.append(path_prefix)
        # Escape LIKE metacharacters so the prefix matches literally — paths
        # legitimately contain ``_`` and ``%`` per the schema CHECK regex.
        args.append(_escape_like(path_prefix))
        where += f" AND (path = ${len(args) - 1} OR path LIKE ${len(args)} || '%')"
    order_sql = "path ASC" if order_by == "path" else "created_at DESC"
    args.append(limit)
    rows = await conn.fetch(
        f"SELECT * FROM memories WHERE {where} ORDER BY {order_sql} LIMIT ${len(args)}", *args
    )

    memories = [_row_to_memory(r, include_content=False) for r in rows]
    if depth is None:
        return list(memories)

    base = path_prefix.rstrip("/") if path_prefix else ""
    out: list[Memory | MemoryPrefix] = []
    seen_prefixes: set[str] = set()
    for memory in memories:
        rest = memory.path[len(base) :] if memory.path.startswith(base) else memory.path
        # rest looks like "/segment/segment/file"; strip the leading "/" and split
        parts = rest.lstrip("/").split("/")
        if len(parts) <= depth:
            out.append(memory)
            continue
        prefix_path = base + "/" + "/".join(parts[:depth]) + "/"
        if prefix_path in seen_prefixes:
            continue
        seen_prefixes.add(prefix_path)
        out.append(MemoryPrefix(path=prefix_path))
    return out


async def update_memory_with_version(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    store_id: str,
    memory_id: str,
    new_content: str | None,
    new_content_sha256: str | None,
    new_path: str | None,
    precondition_sha256: str | None,
    actor_type: ActorType,
    actor_ref: str,
) -> Memory:
    """Update content and/or path; record a ``modified`` version.

    Precondition (when set) is content-only — renames are unconditional,
    matching Anthropic's wire semantics. If both content and path are None
    the call is a no-op and returns the current row.
    """
    if new_content is None and new_path is None:
        return await get_memory(
            conn, store_id, memory_id, include_content=False, account_id=account_id
        )

    next_path_for_conflict: str | None = None
    try:
        async with conn.transaction():
            cur = await conn.fetchrow(
                "SELECT * FROM memories WHERE memory_store_id = $1 AND id = $2 "
                "AND deleted_at IS NULL FOR UPDATE",
                store_id,
                memory_id,
            )
            if cur is None:
                raise NotFoundError(
                    f"memory {memory_id} not found in store {store_id}",
                    detail={"id": memory_id, "memory_store_id": store_id},
                )

            if (
                precondition_sha256 is not None
                and new_content is not None
                and cur["content_sha256"] != precondition_sha256
            ):
                raise MemoryPreconditionFailedError(
                    "precondition content_sha256 failed: content has changed",
                    detail={
                        "expected": precondition_sha256,
                        "actual": cur["content_sha256"],
                    },
                )

            next_content: str = new_content if new_content is not None else cur["content"]
            next_sha: str = (
                new_content_sha256 if new_content_sha256 is not None else cur["content_sha256"]
            )
            next_size: int = len(next_content.encode("utf-8"))
            next_path: str = new_path if new_path is not None else cur["path"]
            next_path_for_conflict = next_path

            seq = await _allocate_version_seq(conn, store_id, account_id=account_id)
            version_id = make_id(MEMORY_VERSION)
            await conn.execute(
                """
                INSERT INTO memory_versions
                    (id, memory_store_id, memory_id, seq, operation, path,
                     content, content_sha256, content_size_bytes,
                     created_by_type, created_by_ref, account_id)
                VALUES ($1, $2, $3, $4, 'modified', $5, $6, $7, $8, $9, $10, $11)
                """,
                version_id,
                store_id,
                memory_id,
                seq,
                next_path,
                next_content,
                next_sha,
                next_size,
                actor_type,
                actor_ref,
                account_id,
            )

            row = await conn.fetchrow(
                "UPDATE memories SET content = $1, content_sha256 = $2, "
                "content_size_bytes = $3, path = $4, current_version_id = $5, "
                "updated_at = now() "
                "WHERE memory_store_id = $6 AND id = $7 RETURNING *",
                next_content,
                next_sha,
                next_size,
                next_path,
                version_id,
                store_id,
                memory_id,
            )
    except asyncpg.UniqueViolationError as exc:
        assert next_path_for_conflict is not None
        existing = await conn.fetchrow(
            "SELECT id FROM memories WHERE memory_store_id = $1 AND path = $2 "
            "AND id != $3 AND deleted_at IS NULL",
            store_id,
            next_path_for_conflict,
            memory_id,
        )
        conflicting_id = existing["id"] if existing is not None else None
        raise MemoryPathConflictError(
            f"path {next_path_for_conflict!r} is already used by {conflicting_id!r}",
            detail={
                "conflicting_memory_id": conflicting_id,
                "conflicting_path": next_path_for_conflict,
            },
        ) from exc
    assert row is not None
    return _row_to_memory(row, include_content=False)


async def delete_memory_with_version(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    store_id: str,
    memory_id: str,
    actor_type: ActorType,
    actor_ref: str,
) -> None:
    """Soft-delete: tombstone version row + ``deleted_at`` on the memory."""
    async with conn.transaction():
        cur = await conn.fetchrow(
            "SELECT path FROM memories WHERE memory_store_id = $1 AND id = $2 "
            "AND deleted_at IS NULL FOR UPDATE",
            store_id,
            memory_id,
        )
        if cur is None:
            raise NotFoundError(
                f"memory {memory_id} not found in store {store_id}",
                detail={"id": memory_id, "memory_store_id": store_id},
            )

        seq = await _allocate_version_seq(conn, store_id, account_id=account_id)
        version_id = make_id(MEMORY_VERSION)
        await conn.execute(
            """
            INSERT INTO memory_versions
                (id, memory_store_id, memory_id, seq, operation, path,
                 created_by_type, created_by_ref, account_id)
            VALUES ($1, $2, $3, $4, 'deleted', $5, $6, $7, $8)
            """,
            version_id,
            store_id,
            memory_id,
            seq,
            cur["path"],
            actor_type,
            actor_ref,
            account_id,
        )

        await conn.execute(
            "UPDATE memories SET deleted_at = now(), updated_at = now() "
            "WHERE memory_store_id = $1 AND id = $2",
            store_id,
            memory_id,
        )


# Versions ─────────────────────────────────────────────────────────────────


async def list_memory_versions(
    conn: asyncpg.Connection[Any],
    store_id: str,
    *,
    account_id: str,
    memory_id: str | None = None,
    limit: int = 100,
) -> list[MemoryVersion]:
    args: list[Any] = [store_id, account_id]
    where = "memory_store_id = $1 AND account_id = $2"
    if memory_id is not None:
        args.append(memory_id)
        where += f" AND memory_id = ${len(args)}"
    args.append(limit)
    # ``seq DESC`` is the load-bearing tiebreaker: ``created_at`` defaults
    # to transaction-start ``now()``, so rows written in the same
    # transaction (any bulk-edit flow, e.g. multiple ``update_memory``
    # calls under one HTTP request) share ``created_at`` to the
    # microsecond. The ``UNIQUE (memory_store_id, seq)`` constraint makes
    # ``seq`` per-store-monotonic and unambiguous, and it's allocated in
    # write order by ``_allocate_version_seq`` — so ``seq DESC`` agrees
    # with "newest first" within the tied group.
    rows = await conn.fetch(
        f"SELECT * FROM memory_versions WHERE {where} "
        f"ORDER BY created_at DESC, seq DESC LIMIT ${len(args)}",
        *args,
    )
    return [_row_to_memory_version(r, include_content=False) for r in rows]


async def get_memory_version(
    conn: asyncpg.Connection[Any],
    store_id: str,
    version_id: str,
    *,
    account_id: str,
) -> MemoryVersion:
    row = await conn.fetchrow(
        "SELECT * FROM memory_versions WHERE memory_store_id = $1 AND id = $2 AND account_id = $3",
        store_id,
        version_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"memory version {version_id} not found",
            detail={"id": version_id, "memory_store_id": store_id},
        )
    return _row_to_memory_version(row, include_content=True)


async def redact_memory_version(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    store_id: str,
    version_id: str,
    actor_type: ActorType,
    actor_ref: str,
) -> MemoryVersion:
    """Strip content fields from a version while keeping the audit trail.

    Rejects redacting the current head of a live (non-deleted) memory:
    write a new version first, or delete the parent memory.
    """
    async with conn.transaction():
        ver = await conn.fetchrow(
            "SELECT * FROM memory_versions "
            "WHERE memory_store_id = $1 AND id = $2 AND account_id = $3 "
            "FOR UPDATE",
            store_id,
            version_id,
            account_id,
        )
        if ver is None:
            raise NotFoundError(
                f"memory version {version_id} not found",
                detail={"id": version_id, "memory_store_id": store_id},
            )

        head_check = await conn.fetchrow(
            "SELECT 1 FROM memories WHERE memory_store_id = $1 AND id = $2 "
            "AND current_version_id = $3 AND account_id = $4 "
            "AND deleted_at IS NULL",
            store_id,
            ver["memory_id"],
            version_id,
            account_id,
        )
        if head_check is not None:
            raise ConflictError(
                "this version is the live head; write a new version first, "
                "or delete the memory to make all versions redactable",
                detail={"id": version_id, "memory_id": ver["memory_id"]},
            )

        if ver["redacted_at"] is not None:
            return _row_to_memory_version(ver, include_content=False)

        row = await conn.fetchrow(
            "UPDATE memory_versions SET path = NULL, content = NULL, "
            "content_sha256 = NULL, content_size_bytes = NULL, "
            "redacted_at = now(), redacted_by_type = $1, redacted_by_ref = $2 "
            "WHERE memory_store_id = $3 AND id = $4 RETURNING *",
            actor_type,
            actor_ref,
            store_id,
            version_id,
        )
    assert row is not None
    return _row_to_memory_version(row, include_content=False)


# Session attachment ───────────────────────────────────────────────────────


async def attach_memory_stores_to_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resources: list[MemoryStoreResource],
    *,
    account_id: str,
) -> None:
    """Insert ``session_memory_stores`` rows for each resource, snapshotting
    name + description from the parent store at attach time. Validates that
    every referenced store exists and is non-archived; rejects duplicate
    snapshotted names (mount-path collision)."""
    if not resources:
        return
    seen_names: set[str] = set()
    for rank, res in enumerate(resources):
        store = await get_memory_store(
            conn, res.memory_store_id, allow_archived=False, account_id=account_id
        )
        if store.name in seen_names:
            raise ConflictError(
                f"two attached memory stores share the name {store.name!r}; "
                "rename one before attaching",
                detail={
                    "memory_store_id": res.memory_store_id,
                    "conflicting_name": store.name,
                },
            )
        seen_names.add(store.name)
        await conn.execute(
            """
            INSERT INTO session_memory_stores
                (session_id, memory_store_id, rank, access, instructions,
                 name_at_attach, description_at_attach, account_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            session_id,
            res.memory_store_id,
            rank,
            res.access,
            res.instructions,
            store.name,
            store.description,
            account_id,
        )


def _row_to_memory_store_echo(row: asyncpg.Record) -> MemoryStoreResourceEcho:
    return MemoryStoreResourceEcho(
        memory_store_id=row["memory_store_id"],
        access=row["access"],
        instructions=row["instructions"],
        name=row["name_at_attach"],
        description=row["description_at_attach"],
        mount_path=f"/mnt/memory/{row['name_at_attach']}",
    )


async def list_session_memory_store_echoes(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[MemoryStoreResourceEcho]:
    rows = await conn.fetch(
        "SELECT * FROM session_memory_stores WHERE session_id = $1 AND account_id = $2 ORDER BY rank",
        session_id,
        account_id,
    )
    return [_row_to_memory_store_echo(r) for r in rows]


# GitHub repository attachments ────────────────────────────────────────────


def _row_to_github_repo_echo(row: asyncpg.Record) -> GithubRepositoryResourceEcho:
    return GithubRepositoryResourceEcho(
        id=row["id"],
        url=row["repo_url"],
        mount_path=row["mount_path"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        git_user_name=row["git_user_name"],
        git_user_email=row["git_user_email"],
    )


async def attach_github_repos_to_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    entries: list[tuple[str, str, EncryptedBlob, str | None, str | None]],
    *,
    account_id: str,
) -> None:
    """Insert pre-encrypted github_repository attachments for a session.

    ``entries`` is ``(repo_url, mount_path, encrypted_token,
    git_user_name, git_user_email)`` tuples in rank order. Encryption is
    the caller's responsibility (service layer holds the CryptoBox).
    Uniqueness on (session_id, mount_path) is enforced by the partial
    unique index — a duplicate raises asyncpg's
    ``UniqueViolationError`` which the service layer maps to a 4xx.
    """
    for rank, (repo_url, mount_path, blob, git_user_name, git_user_email) in enumerate(entries):
        rid = make_id(GITHUB_REPOSITORY)
        await conn.execute(
            """
            INSERT INTO session_github_repositories
                (id, session_id, rank, repo_url, mount_path, ciphertext, nonce,
                 git_user_name, git_user_email, account_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            rid,
            session_id,
            rank,
            repo_url,
            mount_path,
            blob.ciphertext,
            blob.nonce,
            git_user_name,
            git_user_email,
            account_id,
        )


async def list_session_github_repo_echoes(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[GithubRepositoryResourceEcho]:
    rows = await conn.fetch(
        "SELECT * FROM session_github_repositories WHERE session_id = $1 AND account_id = $2 ORDER BY rank",
        session_id,
        account_id,
    )
    return [_row_to_github_repo_echo(r) for r in rows]


async def batch_list_session_memory_store_echoes(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
) -> dict[str, list[MemoryStoreResourceEcho]]:
    """Batch-fetch memory-store echoes for multiple sessions, keyed by session_id."""
    if not session_ids:
        return {}
    rows = await conn.fetch(
        "SELECT session_id, memory_store_id, access, instructions, "
        "name_at_attach, description_at_attach "
        "FROM session_memory_stores "
        "WHERE session_id = ANY($1) AND account_id = $2 "
        "ORDER BY session_id, rank",
        session_ids,
        account_id,
    )
    result: dict[str, list[MemoryStoreResourceEcho]] = {sid: [] for sid in session_ids}
    for r in rows:
        result[r["session_id"]].append(_row_to_memory_store_echo(r))
    return result


async def batch_list_session_github_repo_echoes(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
) -> dict[str, list[GithubRepositoryResourceEcho]]:
    """Batch-fetch github-repository echoes for multiple sessions, keyed by session_id."""
    if not session_ids:
        return {}
    rows = await conn.fetch(
        "SELECT session_id, id, repo_url, mount_path, created_at, updated_at, "
        "git_user_name, git_user_email "
        "FROM session_github_repositories "
        "WHERE session_id = ANY($1) AND account_id = $2 "
        "ORDER BY session_id, rank",
        session_ids,
        account_id,
    )
    result: dict[str, list[GithubRepositoryResourceEcho]] = {sid: [] for sid in session_ids}
    for r in rows:
        result[r["session_id"]].append(_row_to_github_repo_echo(r))
    return result


async def get_session_github_repo(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resource_id: str,
    *,
    account_id: str,
) -> GithubRepositoryResourceEcho:
    row = await conn.fetchrow(
        "SELECT * FROM session_github_repositories "
        "WHERE session_id = $1 AND id = $2 AND account_id = $3",
        session_id,
        resource_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"github_repository resource {resource_id} not found on session {session_id}",
            detail={"session_id": session_id, "resource_id": resource_id},
        )
    return _row_to_github_repo_echo(row)


async def get_session_github_repo_with_blob(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resource_id: str,
    *,
    account_id: str,
) -> tuple[GithubRepositoryResourceEcho, EncryptedBlob]:
    """Read view + encrypted token blob, for the rotation path which needs
    both."""
    row = await conn.fetchrow(
        "SELECT * FROM session_github_repositories "
        "WHERE session_id = $1 AND id = $2 AND account_id = $3",
        session_id,
        resource_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"github_repository resource {resource_id} not found on session {session_id}",
            detail={"session_id": session_id, "resource_id": resource_id},
        )
    return _row_to_github_repo_echo(row), EncryptedBlob(
        ciphertext=row["ciphertext"], nonce=row["nonce"]
    )


async def update_session_github_repo_blob(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resource_id: str,
    blob: EncryptedBlob,
    *,
    account_id: str,
    identity: tuple[str | None, str | None] | None = None,
) -> GithubRepositoryResourceEcho:
    """Replace the encrypted token blob and bump ``updated_at``.

    ``url`` and ``mount_path`` are immutable to match CMA's behavior
    (verified by API probe — PUT returns 405, DELETE returns 400, only
    POST with ``{authorization_token}`` is accepted).  ``identity`` is
    ``None`` to preserve the existing ``git_user_name`` /
    ``git_user_email`` (the common token-only rotation), or a
    ``(name, email)`` tuple to replace both fields atomically — either
    component may itself be ``None`` to clear that column.
    """
    if identity is None:
        row = await conn.fetchrow(
            """
            UPDATE session_github_repositories
            SET ciphertext = $1, nonce = $2, updated_at = now()
            WHERE session_id = $3 AND id = $4 AND account_id = $5
            RETURNING *
            """,
            blob.ciphertext,
            blob.nonce,
            session_id,
            resource_id,
            account_id,
        )
    else:
        git_user_name, git_user_email = identity
        row = await conn.fetchrow(
            """
            UPDATE session_github_repositories
            SET ciphertext = $1, nonce = $2,
                git_user_name = $3, git_user_email = $4,
                updated_at = now()
            WHERE session_id = $5 AND id = $6 AND account_id = $7
            RETURNING *
            """,
            blob.ciphertext,
            blob.nonce,
            git_user_name,
            git_user_email,
            session_id,
            resource_id,
            account_id,
        )
    if row is None:
        raise NotFoundError(
            f"github_repository resource {resource_id} not found on session {session_id}",
            detail={"session_id": session_id, "resource_id": resource_id},
        )
    return _row_to_github_repo_echo(row)


async def delete_session_github_repos(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> None:
    """Delete all github_repository attachments for a session.

    Used by the full-list-replace path on session update — paired with
    a re-insert via :func:`attach_github_repos_to_session` inside the
    same transaction.
    """
    await conn.execute(
        "DELETE FROM session_github_repositories WHERE session_id = $1 AND account_id = $2",
        session_id,
        account_id,
    )


async def delete_session_memory_store(
    conn: asyncpg.Connection[Any],
    session_id: str,
    memory_store_id: str,
    *,
    account_id: str,
) -> None:
    """Detach a single memory store from a session by ``memory_store_id``.

    The granular remove-one path (#270). ``memory_versions`` rows are
    never touched (never-delete) — only the ``session_memory_stores``
    binding row goes away. Raises :class:`NotFoundError` if no row
    matched (unknown id, or wrong account).
    """
    result = await conn.execute(
        "DELETE FROM session_memory_stores "
        "WHERE session_id = $1 AND memory_store_id = $2 AND account_id = $3",
        session_id,
        memory_store_id,
        account_id,
    )
    if result == "DELETE 0":
        raise NotFoundError(
            f"memory_store {memory_store_id} not attached to session {session_id}",
            detail={"session_id": session_id, "memory_store_id": memory_store_id},
        )


async def delete_session_github_repo(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resource_id: str,
    *,
    account_id: str,
) -> None:
    """Detach a single github_repository attachment by its row id (#270).

    Single-row variant of :func:`delete_session_github_repos`. Raises
    :class:`NotFoundError` if no row matched.
    """
    result = await conn.execute(
        "DELETE FROM session_github_repositories "
        "WHERE session_id = $1 AND id = $2 AND account_id = $3",
        session_id,
        resource_id,
        account_id,
    )
    if result == "DELETE 0":
        raise NotFoundError(
            f"github_repository resource {resource_id} not found on session {session_id}",
            detail={"session_id": session_id, "resource_id": resource_id},
        )


async def acquire_session_resources_lock(
    conn: asyncpg.Connection[Any],
    session_id: str,
) -> None:
    """Per-session transaction-scoped advisory lock for resource cap
    enforcement (#270).

    Held for the duration of the surrounding transaction; serializes
    concurrent count-check + INSERT pairs across workers so the per-type
    resource caps (``MAX_STORES_PER_SESSION`` / ``MAX_REPOS_PER_SESSION``)
    are contractual instead of approximate. Mirrors the idiom of
    :func:`aios.db.queries.acquire_account_triggers_lock`.
    """
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        f"resources:{session_id}",
    )


async def list_session_memory_store_ranks(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[int]:
    """Return the in-use ranks for a session's memory-store attachments.

    The granular add-one path (#270) needs the used ranks to pick the
    lowest free rank in ``0..7`` (memory rank has a
    ``CHECK (rank BETWEEN 0 AND 7)``), so a naive ``max(rank)+1`` after a
    low-rank delete doesn't violate the bound.
    """
    rows = await conn.fetch(
        "SELECT rank FROM session_memory_stores WHERE session_id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    return [r["rank"] for r in rows]


async def insert_session_memory_store(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    memory_store_id: str,
    rank: int,
    access: Access,
    instructions: str,
    name_at_attach: str,
    description_at_attach: str,
    account_id: str,
) -> None:
    """Insert a single ``session_memory_stores`` row at an explicit rank.

    Single-row variant of :func:`attach_memory_stores_to_session` for the
    granular add-one path (#270); the caller resolves the snapshotted
    name/description and the lowest-free rank.
    """
    await conn.execute(
        """
        INSERT INTO session_memory_stores
            (session_id, memory_store_id, rank, access, instructions,
             name_at_attach, description_at_attach, account_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        session_id,
        memory_store_id,
        rank,
        access,
        instructions,
        name_at_attach,
        description_at_attach,
        account_id,
    )


async def list_session_github_repo_ranks(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[int]:
    """Return the in-use ranks for a session's github_repository
    attachments (granular add-one rank assignment, #270)."""
    rows = await conn.fetch(
        "SELECT rank FROM session_github_repositories WHERE session_id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    return [r["rank"] for r in rows]


async def insert_session_github_repo(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    rank: int,
    repo_url: str,
    mount_path: str,
    blob: EncryptedBlob,
    git_user_name: str | None,
    git_user_email: str | None,
    account_id: str,
) -> GithubRepositoryResourceEcho:
    """Insert a single ``session_github_repositories`` row at an explicit
    rank and return its echo.

    Single-row variant of :func:`attach_github_repos_to_session` for the
    granular add-one path (#270). A ``(session_id, mount_path)`` collision
    surfaces as :class:`asyncpg.UniqueViolationError`, which the service
    layer maps to a 4xx (same as the bulk attach).
    """
    rid = make_id(GITHUB_REPOSITORY)
    row = await conn.fetchrow(
        """
        INSERT INTO session_github_repositories
            (id, session_id, rank, repo_url, mount_path, ciphertext, nonce,
             git_user_name, git_user_email, account_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING *
        """,
        rid,
        session_id,
        rank,
        repo_url,
        mount_path,
        blob.ciphertext,
        blob.nonce,
        git_user_name,
        git_user_email,
        account_id,
    )
    assert row is not None
    return _row_to_github_repo_echo(row)
