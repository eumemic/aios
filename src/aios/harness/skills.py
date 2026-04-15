"""System prompt augmentation and filesystem provisioning for skills.

Two responsibilities:

1. **Prompt augmentation** — append skill metadata (name, description,
   filesystem location) to the agent's system prompt. This is the
   "Level 1" progressive disclosure: ~100 tokens per skill, always
   present in context.

2. **File provisioning** — write skill files to the session's workspace
   directory on the host. The workspace is bind-mounted into the
   container at ``/workspace``, so files appear at
   ``/workspace/skills/{directory}/``. The model reads ``SKILL.md``
   on demand via the ``read`` tool ("Level 2" disclosure).

Both functions are called from :func:`run_session_step` before the
model inference call.
"""

from __future__ import annotations

from aios.db import queries
from aios.logging import get_logger
from aios.models.skills import SkillVersion
from aios.sandbox.volumes import ensure_workspace_path

log = get_logger("aios.harness.skills")


async def _load_workspace_path(session_id: str) -> str:
    """Load the workspace volume path for a session from the DB."""
    from aios.harness import runtime

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        return await queries.get_session_workspace_path(conn, session_id)


# ── system prompt augmentation ─────────────────────────────────────────────

_SKILLS_POLICY = """\
# Skills usage policy
Before writing any code or using any execution tools, you must:
- Scan the user message for trigger words listed in each skill's description
- If any trigger word is found, immediately use the read tool to read the corresponding SKILL.md file
- Always check available skills before starting work on a new task
- Follow all instructions in SKILL.md exactly when a skill is activated
- Skills are available under /workspace/skills/"""


def build_skills_system_block(skill_versions: list[SkillVersion]) -> str:
    """Build the ``<available_skills>`` block for the system prompt.

    Returns an empty string if no skills are provided.
    """
    if not skill_versions:
        return ""

    lines = [_SKILLS_POLICY, "", "<available_skills>"]
    for sv in skill_versions:
        lines.append("<skill>")
        lines.append(f"  <name>{sv.name}</name>")
        lines.append(f"  <description>{sv.description}</description>")
        lines.append(f"  <location>/workspace/skills/{sv.directory}/SKILL.md</location>")
        lines.append("</skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def augment_system_prompt(
    base_system: str,
    skill_versions: list[SkillVersion],
) -> str:
    """Append skill metadata block to the agent's system prompt.

    No-op if ``skill_versions`` is empty — returns ``base_system``
    unchanged.
    """
    skills_block = build_skills_system_block(skill_versions)
    if not skills_block:
        return base_system
    if base_system:
        return base_system + "\n\n" + skills_block
    return skills_block


# ── filesystem provisioning ────────────────────────────────────────────────


async def provision_skill_files(
    session_id: str,
    skill_versions: list[SkillVersion],
) -> None:
    """Write skill files to the session workspace on the host filesystem.

    The workspace is bind-mounted into the container at ``/workspace``,
    so files written to ``{workspace}/skills/{directory}/`` appear at
    ``/workspace/skills/{directory}/`` inside the container.

    Idempotent: if the ``skills`` directory already exists, skips
    provisioning entirely (files were written on a previous step).
    """
    if not skill_versions:
        return

    raw_path = await _load_workspace_path(session_id)
    workspace = ensure_workspace_path(raw_path)
    skills_dir = workspace / "skills"

    if skills_dir.exists():
        return  # already provisioned

    for sv in skill_versions:
        sv_dir = skills_dir / sv.directory
        for path, content in sv.files.items():
            file_path = sv_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

    log.info(
        "skills.provisioned",
        session_id=session_id,
        skill_count=len(skill_versions),
    )
