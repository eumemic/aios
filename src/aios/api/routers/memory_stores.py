"""HTTP endpoints for memory stores, memories, and memory versions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from aios.api.deps import AccountIdDep, PoolDep
from aios.models.common import ListResponse
from aios.models.memory_stores import (
    Memory,
    MemoryCreate,
    MemoryPrefix,
    MemoryStore,
    MemoryStoreCreate,
    MemoryStoreUpdate,
    MemoryUpdate,
    MemoryVersion,
)
from aios.models.pagination import (
    MAX_PAGE_LIMIT,
    PageLimit,
    page_cursor,
    resolve_page_limit,
)
from aios.services import memory_stores as service

router = APIRouter(prefix="/v1/memory-stores", tags=["memory-stores"])


# ── stores ─────────────────────────────────────────────────────────────────


@router.post("", operation_id="create_memory_store", status_code=status.HTTP_201_CREATED)
async def create_store(
    body: MemoryStoreCreate, pool: PoolDep, account_id: AccountIdDep
) -> MemoryStore:
    """Create a new memory store — a named collection of file-like memories.

    Memories created in this store are mirrored to a per-store host
    directory so that sessions referencing the store can read them through
    the sandbox filesystem.
    """
    return await service.create_store(
        pool,
        name=body.name,
        description=body.description,
        metadata=body.metadata,
        account_id=account_id,
    )


@router.get("", operation_id="list_memory_stores")
async def list_stores(
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    include_archived: bool | None = None,
    limit: PageLimit = None,
) -> ListResponse[MemoryStore]:
    """List memory stores, newest first.

    Unlike most resources, archived stores can be included via
    ``include_archived=true`` (default false). First page: optional
    ``include_archived`` + ``?limit=``. Subsequent pages:
    ``?cursor=<next_cursor>``. The default limit is 100 since stores are few.
    """
    st = page_cursor(cursor, {"include_archived": include_archived, "limit": limit})
    after = str(st.cursor) if st is not None else None
    page_limit = resolve_page_limit(st, limit, default=100)
    archived = (
        bool(st.filters.get("include_archived")) if st is not None else bool(include_archived)
    )
    items = await service.list_stores(
        pool,
        include_archived=archived,
        limit=page_limit + 1,
        after=after,
        account_id=account_id,
    )
    return ListResponse[MemoryStore].paginate(
        items, page_limit, cursor=lambda x: x.id, filters={"include_archived": archived}
    )


@router.get("/{store_id}", operation_id="get_memory_store")
async def get_store(store_id: str, pool: PoolDep, account_id: AccountIdDep) -> MemoryStore:
    """Fetch one memory store by id."""
    return await service.get_store(pool, store_id, account_id=account_id)


@router.post("/{store_id}", operation_id="update_memory_store")
async def update_store(
    store_id: str, body: MemoryStoreUpdate, pool: PoolDep, account_id: AccountIdDep
) -> MemoryStore:
    """Update a memory store's ``name``, ``description``, or ``metadata``.

    Rejects with ``MemoryStoreArchivedError`` if the store is archived —
    archived stores are read-only. Treat as partial update on the listed
    fields.
    """
    return await service.update_store(
        pool,
        store_id,
        name=body.name,
        description=body.description,
        metadata=body.metadata,
        account_id=account_id,
    )


@router.post(
    "/{store_id}/archive",
    operation_id="archive_memory_store",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def archive_store(store_id: str, pool: PoolDep, account_id: AccountIdDep) -> MemoryStore:
    """Archive a memory store: hides from default lists, makes it read-only.

    The store and its memories persist; sessions can still resolve memory
    content. Subsequent ``update_memory_store`` calls fail until
    un-archived (no API surface for that currently). Use
    ``delete_memory_store`` for full removal.
    """
    return await service.archive_store(pool, store_id, account_id=account_id)


@router.delete(
    "/{store_id}",
    operation_id="delete_memory_store",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_store(store_id: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Hard-delete a memory store, all its memories, and its host mirror.

    Cascade-deletes memories and versions in the database. After the DB
    transaction commits, the per-store host mirror directory is removed
    (best-effort — a missing dir is fine, indicates no session ever
    provisioned for this store). Returns 204.
    """
    await service.delete_store(pool, store_id, account_id=account_id)


# ── memories ────────────────────────────────────────────────────────────────


@router.post(
    "/{store_id}/memories",
    operation_id="create_memory",
    status_code=status.HTTP_201_CREATED,
)
async def create_memory(
    store_id: str, body: MemoryCreate, pool: PoolDep, account_id: AccountIdDep
) -> Memory:
    """Create a memory at the given path within a store.

    The content is also mirrored to the store's host directory so that
    sandboxed sessions can read it through the filesystem. Creates an
    initial version (every memory mutation creates a version).
    """
    return await service.create_memory(
        pool,
        store_id=store_id,
        path=body.path,
        content=body.content,
        actor=service.ApiActor(),
        account_id=account_id,
    )


@router.get("/{store_id}/memories", operation_id="list_memories")
async def list_memories(
    store_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    path_prefix: str | None = None,
    order_by: str = "created_at",
    depth: int | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = 100,
) -> ListResponse[Memory | MemoryPrefix]:
    """List memories in a store, optionally filtered and grouped by path.

    ``path_prefix`` is a literal prefix match on the memory path. ``depth``
    groups deeper paths into ``MemoryPrefix`` entries (directory-style
    listings) — entries past the depth boundary are collapsed into a
    single prefix entry per shared directory. ``order_by`` accepts
    ``created_at`` (default) or ``path``. ``limit`` caps the raw-row
    fetch (cursor pagination not yet supported; use ``path_prefix`` to
    narrow scope when a store has thousands of memories).
    """
    items = await service.list_memories(
        pool,
        store_id,
        path_prefix=path_prefix,
        order_by=order_by,
        depth=depth,
        limit=limit,
        account_id=account_id,
    )
    # ``has_more`` signals the SQL cap was hit; depth aggregation may have
    # collapsed those raw rows into fewer response entries, so compare the
    # underlying memory count (entries that aren't MemoryPrefix) plus
    # collapsed prefix groups against the limit.
    raw_count = sum(1 for it in items if not isinstance(it, MemoryPrefix))
    has_more = raw_count == limit
    return ListResponse[Memory | MemoryPrefix](data=items, has_more=has_more)


@router.get("/{store_id}/memories/{memory_id}", operation_id="get_memory")
async def get_memory(
    store_id: str, memory_id: str, pool: PoolDep, account_id: AccountIdDep
) -> Memory:
    """Fetch one memory by id, including its current content."""
    return await service.get_memory(
        pool, store_id, memory_id, include_content=True, account_id=account_id
    )


@router.post("/{store_id}/memories/{memory_id}", operation_id="update_memory")
async def update_memory(
    store_id: str,
    memory_id: str,
    body: MemoryUpdate,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> Memory:
    """Update a memory's content and/or path. Creates a new version.

    Optionally honors a ``precondition.content_sha256`` for optimistic
    concurrency — if the current content's SHA-256 differs, the update
    fails with a precondition error rather than overwriting. The host
    mirror is updated to match: renames delete the old path and write the
    new one.
    """
    return await service.update_memory(
        pool,
        store_id=store_id,
        memory_id=memory_id,
        new_content=body.content,
        new_path=body.path,
        precondition_sha256=(
            body.precondition.content_sha256 if body.precondition is not None else None
        ),
        actor=service.ApiActor(),
        account_id=account_id,
    )


@router.delete(
    "/{store_id}/memories/{memory_id}",
    operation_id="delete_memory",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_memory(
    store_id: str, memory_id: str, pool: PoolDep, account_id: AccountIdDep
) -> None:
    """Soft-delete a memory and remove it from the host mirror.

    Sets ``deleted_at`` on the memory and appends a ``deleted`` tombstone
    version. The row and version history persist; live reads filter on
    ``deleted_at IS NULL`` so the memory becomes invisible. Returns 204.
    """
    await service.delete_memory(
        pool,
        store_id=store_id,
        memory_id=memory_id,
        actor=service.ApiActor(),
        account_id=account_id,
    )


# ── versions ────────────────────────────────────────────────────────────────


@router.get("/{store_id}/memory-versions", operation_id="list_memory_versions")
async def list_versions(
    store_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    memory_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = 100,
) -> ListResponse[MemoryVersion]:
    """List memory versions in a store, newest first.

    Optional ``memory_id`` filters to a single memory's version history.
    Without the filter, returns versions across all memories in the store
    (useful for audit). No cursor pagination; bumps default limit to 100.
    """
    items = await service.list_versions(
        pool, store_id, memory_id=memory_id, limit=limit, account_id=account_id
    )
    return ListResponse[MemoryVersion](data=items)


@router.get("/{store_id}/memory-versions/{version_id}", operation_id="get_memory_version")
async def get_version(
    store_id: str, version_id: str, pool: PoolDep, account_id: AccountIdDep
) -> MemoryVersion:
    """Fetch one historical memory version by id."""
    return await service.get_version(pool, store_id, version_id, account_id=account_id)


@router.post(
    "/{store_id}/memory-versions/{version_id}/redact",
    operation_id="redact_memory_version",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def redact_version(
    store_id: str, version_id: str, pool: PoolDep, account_id: AccountIdDep
) -> MemoryVersion:
    """Redact the content of a historical memory version in place.

    The version row persists for audit (with the actor and timestamp) but
    its content field is cleared. Use to scrub sensitive data that was
    previously written into a memory. Live memory content is unaffected
    (only this specific historical version is redacted).
    """
    return await service.redact_version(
        pool,
        store_id=store_id,
        version_id=version_id,
        actor=service.ApiActor(),
        account_id=account_id,
    )
