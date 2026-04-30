"""HTTP endpoints for memory stores, memories, and memory versions."""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
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
from aios.services import memory_stores as service

router = APIRouter(prefix="/v1/memory-stores", tags=["memory-stores"])


# ── stores ─────────────────────────────────────────────────────────────────


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_store(body: MemoryStoreCreate, pool: PoolDep, _auth: AuthDep) -> MemoryStore:
    return await service.create_store(
        pool, name=body.name, description=body.description, metadata=body.metadata
    )


@router.get("")
async def list_stores(
    pool: PoolDep,
    _auth: AuthDep,
    include_archived: bool = False,
    limit: int = 100,
) -> ListResponse[MemoryStore]:
    items = await service.list_stores(pool, include_archived=include_archived, limit=limit)
    return ListResponse[MemoryStore](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{store_id}")
async def get_store(store_id: str, pool: PoolDep, _auth: AuthDep) -> MemoryStore:
    return await service.get_store(pool, store_id)


@router.post("/{store_id}")
async def update_store(
    store_id: str, body: MemoryStoreUpdate, pool: PoolDep, _auth: AuthDep
) -> MemoryStore:
    return await service.update_store(
        pool,
        store_id,
        name=body.name,
        description=body.description,
        metadata=body.metadata,
    )


@router.post("/{store_id}/archive")
async def archive_store(store_id: str, pool: PoolDep, _auth: AuthDep) -> MemoryStore:
    return await service.archive_store(pool, store_id)


@router.delete("/{store_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_store(store_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.delete_store(pool, store_id)


# ── memories ────────────────────────────────────────────────────────────────


@router.post("/{store_id}/memories", status_code=status.HTTP_201_CREATED)
async def create_memory(store_id: str, body: MemoryCreate, pool: PoolDep, _auth: AuthDep) -> Memory:
    return await service.create_memory(
        pool,
        store_id=store_id,
        path=body.path,
        content=body.content,
        actor=service.ApiActor(),
    )


@router.get("/{store_id}/memories")
async def list_memories(
    store_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    path_prefix: str | None = None,
    order_by: str = "created_at",
    depth: int | None = None,
) -> ListResponse[Memory | MemoryPrefix]:
    items = await service.list_memories(
        pool,
        store_id,
        path_prefix=path_prefix,
        order_by=order_by,
        depth=depth,
    )
    return ListResponse[Memory | MemoryPrefix](data=items)


@router.get("/{store_id}/memories/{memory_id}")
async def get_memory(store_id: str, memory_id: str, pool: PoolDep, _auth: AuthDep) -> Memory:
    return await service.get_memory(pool, store_id, memory_id, include_content=True)


@router.post("/{store_id}/memories/{memory_id}")
async def update_memory(
    store_id: str,
    memory_id: str,
    body: MemoryUpdate,
    pool: PoolDep,
    _auth: AuthDep,
) -> Memory:
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
    )


@router.delete("/{store_id}/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(store_id: str, memory_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.delete_memory(
        pool,
        store_id=store_id,
        memory_id=memory_id,
        actor=service.ApiActor(),
    )


# ── versions ────────────────────────────────────────────────────────────────


@router.get("/{store_id}/memory-versions")
async def list_versions(
    store_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    memory_id: str | None = None,
    limit: int = 100,
) -> ListResponse[MemoryVersion]:
    items = await service.list_versions(pool, store_id, memory_id=memory_id, limit=limit)
    return ListResponse[MemoryVersion](data=items)


@router.get("/{store_id}/memory-versions/{version_id}")
async def get_version(
    store_id: str, version_id: str, pool: PoolDep, _auth: AuthDep
) -> MemoryVersion:
    return await service.get_version(pool, store_id, version_id)


@router.post("/{store_id}/memory-versions/{version_id}/redact")
async def redact_version(
    store_id: str, version_id: str, pool: PoolDep, _auth: AuthDep
) -> MemoryVersion:
    return await service.redact_version(
        pool,
        store_id=store_id,
        version_id=version_id,
        actor=service.ApiActor(),
    )
