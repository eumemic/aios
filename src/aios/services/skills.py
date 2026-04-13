"""Business logic for skills.

Skills are versioned knowledge bundles attached to agents. Each skill
must include a ``SKILL.md`` file with YAML frontmatter containing at
least a ``name`` field. The ``name`` and ``description`` are extracted
from the frontmatter; the ``directory`` is inferred from the file path
prefix.
"""

from __future__ import annotations

import json
import re
from typing import Any

import asyncpg

from aios.db import queries
from aios.errors import ValidationError
from aios.models.skills import AgentSkillRef, Skill, SkillVersion

MAX_SKILLS_PER_AGENT = 20


# ── frontmatter parsing ────────────────────────────────────────────────────


def parse_skill_md(content: str) -> tuple[str, str]:
    """Extract ``(name, description)`` from SKILL.md YAML frontmatter.

    Supports a minimal subset: flat ``key: value`` lines between ``---``
    delimiters. No PyYAML dependency needed.

    Raises :class:`ValidationError` if frontmatter is missing or ``name``
    is absent.
    """
    stripped = content.lstrip("\n")
    if not stripped.startswith("---"):
        raise ValidationError(
            "SKILL.md must start with YAML frontmatter (---)",
            detail={"hint": "Add --- delimiters around name/description fields"},
        )
    # Split on the second --- to get the frontmatter block.
    parts = stripped.split("---", 2)
    if len(parts) < 3:
        raise ValidationError(
            "SKILL.md frontmatter is missing closing ---",
            detail={"hint": "Ensure frontmatter is delimited by --- on both sides"},
        )
    fm_block = parts[1]
    fields: dict[str, str] = {}
    for line in fm_block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-z_][a-z0-9_-]*)\s*:\s*(.*)", line, re.IGNORECASE)
        if m:
            fields[m.group(1).lower()] = m.group(2).strip().strip("'\"")

    name = fields.get("name", "").strip()
    if not name:
        raise ValidationError(
            "SKILL.md frontmatter must contain a 'name' field",
            detail={"fields_found": list(fields.keys())},
        )
    _validate_skill_name(name)

    description = fields.get("description", "").strip()

    # If no description in frontmatter, use the first non-empty body line.
    if not description:
        body = parts[2].strip()
        for line in body.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                description = line
                break

    return name, description


def _validate_skill_name(name: str) -> None:
    """Validate skill name constraints (matching Anthropic's rules)."""
    if len(name) > 64:
        raise ValidationError(
            f"skill name must be at most 64 characters, got {len(name)}",
            detail={"name": name},
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValidationError(
            "skill name must contain only lowercase letters, numbers, and hyphens, "
            "and start with a letter or number",
            detail={"name": name},
        )


def _extract_skill_metadata(
    files: dict[str, str],
) -> tuple[str, str, str, dict[str, str]]:
    """Extract directory, name, description, and normalized files from upload.

    The ``files`` dict must contain exactly one ``{directory}/SKILL.md``
    entry. Returns ``(directory, name, description, normalized_files)``
    where normalized_files has the directory prefix stripped from paths.
    """
    skill_md_paths = [p for p in files if p.endswith("/SKILL.md") or p == "SKILL.md"]
    if len(skill_md_paths) == 0:
        raise ValidationError(
            "files must include a SKILL.md entry (e.g. 'my-skill/SKILL.md')",
            detail={"paths": list(files.keys())},
        )
    if len(skill_md_paths) > 1:
        raise ValidationError(
            "files must include exactly one SKILL.md entry",
            detail={"skill_md_paths": skill_md_paths},
        )

    skill_md_path = skill_md_paths[0]

    if "/" in skill_md_path:
        directory = skill_md_path.rsplit("/", 1)[0].split("/")[0]
    else:
        raise ValidationError(
            "SKILL.md must be inside a directory (e.g. 'my-skill/SKILL.md')",
            detail={"path": skill_md_path},
        )

    name, description = parse_skill_md(files[skill_md_path])

    normalized: dict[str, str] = {}
    prefix = directory + "/"
    for path, content in files.items():
        if path.startswith(prefix):
            normalized[path[len(prefix) :]] = content
        else:
            normalized[path] = content

    return directory, name, description, normalized


# ── skill CRUD ─────────────────────────────────────────────────────────────


async def create_skill(
    pool: asyncpg.Pool[Any],
    *,
    display_title: str,
    files: dict[str, str],
) -> tuple[Skill, SkillVersion]:
    """Create a new skill with its first version.

    Extracts directory, name, and description from the SKILL.md
    frontmatter. Returns ``(skill, version_1)``.
    """
    directory, name, description, normalized = _extract_skill_metadata(files)
    async with pool.acquire() as conn:
        return await queries.insert_skill(
            conn,
            display_title=display_title,
            directory=directory,
            name=name,
            description=description,
            files=normalized,
        )


async def get_skill(pool: asyncpg.Pool[Any], skill_id: str) -> Skill:
    async with pool.acquire() as conn:
        return await queries.get_skill(conn, skill_id)


async def list_skills(
    pool: asyncpg.Pool[Any], *, limit: int = 50, after: str | None = None
) -> list[Skill]:
    async with pool.acquire() as conn:
        return await queries.list_skills(conn, limit=limit, after=after)


async def archive_skill(pool: asyncpg.Pool[Any], skill_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.archive_skill(conn, skill_id)


async def create_skill_version(
    pool: asyncpg.Pool[Any],
    skill_id: str,
    *,
    files: dict[str, str],
) -> SkillVersion:
    """Create a new version of an existing skill."""
    directory, name, description, normalized = _extract_skill_metadata(files)
    async with pool.acquire() as conn:
        return await queries.insert_skill_version(
            conn,
            skill_id=skill_id,
            directory=directory,
            name=name,
            description=description,
            files=normalized,
        )


async def get_skill_version(pool: asyncpg.Pool[Any], skill_id: str, version: int) -> SkillVersion:
    async with pool.acquire() as conn:
        return await queries.get_skill_version(conn, skill_id, version)


async def list_skill_versions(
    pool: asyncpg.Pool[Any],
    skill_id: str,
    *,
    limit: int = 50,
    after: int | None = None,
) -> list[SkillVersion]:
    async with pool.acquire() as conn:
        return await queries.list_skill_versions(conn, skill_id, limit=limit, after=after)


# ── agent skill resolution ────────────────────────────────────────────────


async def resolve_skill_refs(
    pool: asyncpg.Pool[Any],
    refs: list[AgentSkillRef],
) -> list[SkillVersion]:
    """Resolve skill refs to concrete SkillVersion objects.

    Also validates existence: raises ``NotFoundError`` if any skill or
    pinned version doesn't exist. Raises ``ValidationError`` if the
    count exceeds ``MAX_SKILLS_PER_AGENT``.
    """
    if not refs:
        return []
    if len(refs) > MAX_SKILLS_PER_AGENT:
        raise ValidationError(
            f"at most {MAX_SKILLS_PER_AGENT} skills per agent",
            detail={"count": len(refs)},
        )
    async with pool.acquire() as conn:
        return await queries.resolve_skill_refs(conn, refs)


def serialize_skills_for_snapshot(
    refs: list[AgentSkillRef],
    resolved: list[SkillVersion],
) -> str:
    """Build the skills JSON for agent_versions snapshot.

    Replaces null versions with the resolved concrete version numbers
    so the snapshot is fully deterministic.
    """
    snapshot: list[dict[str, Any]] = []
    for ref, sv in zip(refs, resolved, strict=True):
        snapshot.append({"skill_id": ref.skill_id, "version": sv.version})
    return json.dumps(snapshot)
