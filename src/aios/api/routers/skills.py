"""Skill CRUD endpoints with version history."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from aios.api.deps import AccountIdDep, PoolDep
from aios.models.common import ListResponse
from aios.models.skills import Skill, SkillCreate, SkillVersion, SkillVersionCreate
from aios.services import skills as service

router = APIRouter(prefix="/v1/skills", tags=["skills"])


# ── Skill endpoints ────────────────────────────────────────────────────────


@router.post("", operation_id="create_skill", status_code=status.HTTP_201_CREATED)
async def create(body: SkillCreate, pool: PoolDep, account_id: AccountIdDep) -> Skill:
    """Create a new skill from an uploaded file bundle.

    The bundle must include exactly one ``<directory>/SKILL.md`` file with
    YAML frontmatter containing at least a ``name`` field. The skill's
    ``directory``, ``name``, and ``description`` are extracted from that
    frontmatter; subsequent versions reuse this extraction.
    """
    skill, _version = await service.create_skill(
        pool, display_title=body.display_title, files=body.files, account_id=account_id
    )
    return skill


@router.get("", operation_id="list_skills")
async def list_(
    pool: PoolDep,
    account_id: AccountIdDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    after: str | None = None,
) -> ListResponse[Skill]:
    """List skills (latest version of each), newest first, excluding archived.

    Cursor pagination via ``after``.
    """
    items = await service.list_skills(pool, limit=limit + 1, after=after, account_id=account_id)
    return ListResponse[Skill].paginate(items, limit, cursor=lambda x: x.id)


@router.get("/{skill_id}", operation_id="get_skill")
async def get(skill_id: str, pool: PoolDep, account_id: AccountIdDep) -> Skill:
    """Fetch one skill by id, returning the latest version's config."""
    return await service.get_skill(pool, skill_id, account_id=account_id)


@router.delete("/{skill_id}", operation_id="archive_skill", status_code=status.HTTP_204_NO_CONTENT)
async def archive(skill_id: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Archive a skill: sets ``archived_at`` and hides it from default lists.

    The row and version history persist; agents that reference this skill
    by id continue to resolve their pinned versions. There is no API
    surface to un-archive currently.
    """
    await service.archive_skill(pool, skill_id, account_id=account_id)


# ── Skill version endpoints ───────────────────────────────────────────────


@router.post(
    "/{skill_id}/versions",
    operation_id="create_skill_version",
    status_code=status.HTTP_201_CREATED,
)
async def create_version(
    skill_id: str, body: SkillVersionCreate, pool: PoolDep, account_id: AccountIdDep
) -> SkillVersion:
    """Upload a new immutable version of a skill's file bundle.

    The bundle must include the same SKILL.md frontmatter shape as the
    initial create: directory, ``name``, ``description``. Each version is
    a complete snapshot — files not present in the upload are not carried
    over from previous versions.
    """
    return await service.create_skill_version(
        pool, skill_id, files=body.files, account_id=account_id
    )


@router.get("/{skill_id}/versions", operation_id="list_skill_versions")
async def list_versions(
    skill_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    after: int | None = None,
) -> ListResponse[SkillVersion]:
    """List historical versions of a skill, newest first.

    Cursor pagination by version number: pass ``after`` from a previous
    response's ``next_after`` to get the next page. Each version is a
    complete file-bundle snapshot at the time it was created.
    """
    items = await service.list_skill_versions(
        pool, skill_id, limit=limit + 1, after=after, account_id=account_id
    )
    return ListResponse[SkillVersion].paginate(items, limit, cursor=lambda x: str(x.version))


@router.get("/{skill_id}/versions/{version}", operation_id="get_skill_version")
async def get_version(
    skill_id: str, version: int, pool: PoolDep, account_id: AccountIdDep
) -> SkillVersion:
    """Fetch one historical version's file bundle.

    The bundle reflects the skill's state at the time the version was
    written and is unaffected by subsequent versions or archival.
    """
    return await service.get_skill_version(pool, skill_id, version, account_id=account_id)
