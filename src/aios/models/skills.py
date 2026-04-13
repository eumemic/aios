"""Skill resource: versioned knowledge bundles for agents.

Skills are filesystem-based instruction sets that give agents domain-specific
expertise. They use progressive disclosure: only name + description go into
the system prompt (~100 tokens); the full SKILL.md is read on demand via the
``read`` tool from ``/workspace/skills/{directory}/``.

Skills are versioned: every update creates a new immutable version. The
``skills`` table holds the latest metadata; ``skill_versions`` stores the
full history including file content.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# ── Agent skill reference ──────────────────────────────────────────────────


class AgentSkillRef(BaseModel):
    """One entry in an agent's ``skills`` list.

    ``version`` is ``None`` for "latest" (auto-updating — the session uses
    whatever version is current at step time). When an agent version is
    snapshotted, null versions are resolved to the concrete latest version
    at that moment.
    """

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    version: int | None = Field(
        default=None,
        description="Pin to a specific version. Omit or null for latest.",
    )


# ── Create/update request bodies ──────────────────────────────────────────


class SkillCreate(BaseModel):
    """Request body for ``POST /v1/skills``.

    ``files`` must include exactly one ``{directory}/SKILL.md`` entry.
    The server extracts ``name``, ``description``, and ``directory`` from
    the SKILL.md frontmatter and file paths.
    """

    model_config = ConfigDict(extra="forbid")

    display_title: str = Field(min_length=1, max_length=128)
    files: dict[str, str] = Field(
        description=(
            "Skill files as {path: content}. Must include exactly one "
            "{directory}/SKILL.md entry with YAML frontmatter."
        ),
    )


class SkillVersionCreate(BaseModel):
    """Request body for ``POST /v1/skills/{skill_id}/versions``.

    Same file format as :class:`SkillCreate`. The directory, name, and
    description are re-extracted from the new SKILL.md.
    """

    model_config = ConfigDict(extra="forbid")

    files: dict[str, str]


# ── Read views ─────────────────────────────────────────────────────────────


class Skill(BaseModel):
    """Read view of a skill (head row)."""

    id: str
    display_title: str
    latest_version: int
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class SkillVersion(BaseModel):
    """Read view of a specific skill version."""

    skill_id: str
    version: int
    directory: str
    name: str
    description: str
    files: dict[str, str]
    created_at: datetime
