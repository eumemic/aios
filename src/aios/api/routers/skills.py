"""Skill CRUD endpoints with version history."""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.models.common import ListResponse
from aios.models.skills import Skill, SkillCreate, SkillVersion, SkillVersionCreate
from aios.services import skills as service

router = APIRouter(prefix="/v1/skills", tags=["skills"])


# ── Skill endpoints ────────────────────────────────────────────────────────


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: SkillCreate, pool: PoolDep, _auth: AuthDep) -> Skill:
    skill, _version = await service.create_skill(
        pool, display_title=body.display_title, files=body.files
    )
    return skill


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[Skill]:
    items = await service.list_skills(pool, limit=limit, after=after)
    return ListResponse[Skill](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{skill_id}")
async def get(skill_id: str, pool: PoolDep, _auth: AuthDep) -> Skill:
    return await service.get_skill(pool, skill_id)


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive(skill_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_skill(pool, skill_id)


# ── Skill version endpoints ───────────────────────────────────────────────


@router.post("/{skill_id}/versions", status_code=status.HTTP_201_CREATED)
async def create_version(
    skill_id: str, body: SkillVersionCreate, pool: PoolDep, _auth: AuthDep
) -> SkillVersion:
    return await service.create_skill_version(pool, skill_id, files=body.files)


@router.get("/{skill_id}/versions")
async def list_versions(
    skill_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: int | None = None,
) -> ListResponse[SkillVersion]:
    items = await service.list_skill_versions(pool, skill_id, limit=limit, after=after)
    return ListResponse[SkillVersion](
        data=items,
        has_more=len(items) == limit,
        next_after=str(items[-1].version) if items else None,
    )


@router.get("/{skill_id}/versions/{version}")
async def get_version(skill_id: str, version: int, pool: PoolDep, _auth: AuthDep) -> SkillVersion:
    return await service.get_skill_version(pool, skill_id, version)
