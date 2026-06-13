"""Skill queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.db.queries import (
    _archive_scoped,
    _get_scoped,
    _get_versioned,
    _list_scoped,
    _list_versioned,
    parse_jsonb,
)
from aios.errors import (
    NotFoundError,
)
from aios.ids import (
    SKILL,
    make_id,
)
from aios.models.skills import AgentSkillRef, Skill, SkillVersion

# ─── skills ──────────────────────────────────────────────────────────────────


def _row_to_skill(row: asyncpg.Record) -> Skill:
    return Skill(
        id=row["id"],
        display_title=row["display_title"],
        latest_version=row["latest_version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _row_to_skill_version(row: asyncpg.Record) -> SkillVersion:
    files_data = parse_jsonb(row["files"])
    return SkillVersion(
        skill_id=row["skill_id"],
        version=row["version"],
        directory=row["directory"],
        name=row["name"],
        description=row["description"],
        files=files_data,
        created_at=row["created_at"],
    )


async def insert_skill(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    display_title: str,
    directory: str,
    name: str,
    description: str,
    files: dict[str, str],
) -> tuple[Skill, SkillVersion]:
    """Create a skill and its first version atomically.

    Returns ``(skill, version_1)``.
    """
    new_id = make_id(SKILL)
    files_json = json.dumps(files)
    async with conn.transaction():
        skill_row = await conn.fetchrow(
            """
            INSERT INTO skills (id, display_title, latest_version, account_id)
            VALUES ($1, $2, 1, $3)
            RETURNING *
            """,
            new_id,
            display_title,
            account_id,
        )
        assert skill_row is not None
        ver_row = await conn.fetchrow(
            """
            INSERT INTO skill_versions (skill_id, version, directory, name, description, files, account_id)
            VALUES ($1, 1, $2, $3, $4, $5::jsonb, $6)
            RETURNING *
            """,
            new_id,
            directory,
            name,
            description,
            files_json,
            account_id,
        )
        assert ver_row is not None
    return _row_to_skill(skill_row), _row_to_skill_version(ver_row)


async def get_skill(conn: asyncpg.Connection[Any], skill_id: str, *, account_id: str) -> Skill:
    return await _get_scoped(
        conn,
        table="skills",
        id_=skill_id,
        account_id=account_id,
        row=_row_to_skill,
        noun="skill",
    )


async def list_skills(
    conn: asyncpg.Connection[Any], *, account_id: str, limit: int = 50, after: str | None = None
) -> list[Skill]:
    return await _list_scoped(
        conn,
        table="skills",
        account_id=account_id,
        row=_row_to_skill,
        limit=limit,
        after=after,
    )


async def archive_skill(conn: asyncpg.Connection[Any], skill_id: str, *, account_id: str) -> None:
    await _archive_scoped(
        conn,
        table="skills",
        id_=skill_id,
        account_id=account_id,
        noun="skill",
    )


async def insert_skill_version(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    skill_id: str,
    directory: str,
    name: str,
    description: str,
    files: dict[str, str],
) -> SkillVersion:
    """Create a new immutable version for an existing skill.

    Locks the skills row, increments ``latest_version``, inserts the
    version, and updates the head row's ``updated_at``.
    """
    files_json = json.dumps(files)
    async with conn.transaction():
        head = await conn.fetchrow(
            "SELECT latest_version FROM skills WHERE id = $1 AND account_id = $2 FOR UPDATE",
            skill_id,
            account_id,
        )
        if head is None:
            raise NotFoundError(f"skill {skill_id} not found", detail={"id": skill_id})
        new_ver = head["latest_version"] + 1
        ver_row = await conn.fetchrow(
            """
            INSERT INTO skill_versions (skill_id, version, directory, name, description, files, account_id)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            RETURNING *
            """,
            skill_id,
            new_ver,
            directory,
            name,
            description,
            files_json,
            account_id,
        )
        assert ver_row is not None
        await conn.execute(
            "UPDATE skills SET latest_version = $2, updated_at = now() WHERE id = $1 AND account_id = $3",
            skill_id,
            new_ver,
            account_id,
        )
    return _row_to_skill_version(ver_row)


async def get_skill_version(
    conn: asyncpg.Connection[Any],
    skill_id: str,
    version: int,
    *,
    account_id: str,
) -> SkillVersion:
    return await _get_versioned(
        conn,
        table="skill_versions",
        parent_column="skill_id",
        parent_id=skill_id,
        version=version,
        account_id=account_id,
        row=_row_to_skill_version,
        noun="skill",
    )


async def get_latest_skill_version(
    conn: asyncpg.Connection[Any], skill_id: str, *, account_id: str
) -> SkillVersion:
    """Get the latest version of a skill by joining with the head row."""
    row = await conn.fetchrow(
        """
        SELECT sv.* FROM skill_versions sv
        JOIN skills s ON s.id = sv.skill_id AND sv.version = s.latest_version
        WHERE sv.skill_id = $1
          AND sv.account_id = $2
        """,
        skill_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(f"skill {skill_id} has no versions", detail={"skill_id": skill_id})
    return _row_to_skill_version(row)


async def list_skill_versions(
    conn: asyncpg.Connection[Any],
    skill_id: str,
    *,
    account_id: str,
    limit: int = 50,
    after: int | None = None,
) -> list[SkillVersion]:
    """List versions in descending order (newest first)."""
    return await _list_versioned(
        conn,
        table="skill_versions",
        parent_column="skill_id",
        parent_id=skill_id,
        account_id=account_id,
        row=_row_to_skill_version,
        limit=limit,
        after=after,
    )


async def resolve_skill_refs(
    conn: asyncpg.Connection[Any],
    refs: list[AgentSkillRef],
    *,
    account_id: str,
) -> list[SkillVersion]:
    """Resolve a list of skill references to concrete versions.

    For each ref, if ``version`` is ``None``, resolves to the latest
    version. Otherwise fetches the pinned version. Returns versions in
    the same order as the input refs.
    """
    results: list[SkillVersion] = []
    for ref in refs:
        if ref.version is None:
            sv = await get_latest_skill_version(conn, ref.skill_id, account_id=account_id)
        else:
            sv = await get_skill_version(conn, ref.skill_id, ref.version, account_id=account_id)
        results.append(sv)
    return results
