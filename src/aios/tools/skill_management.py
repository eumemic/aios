"""Agent-acting skill builtins — closing the self-improvement loop.

Model-callable tools that let an agent author and revise the versioned skill
bundles attached to it the way a human operator (or the ``aios skills`` CLI)
does. Each is a thin wrapper over the existing skill CRUD service; the agent's
authority is bounded by the executing-session account the harness supplies
(``invoke_builtin(session_id, …)``, ``tools/invoke.py``) — never model input.

This is an exact mirror of the shipped strange-loop workflow builtins
(``src/aios/tools/workflow_management.py``). It adds **no new primitive**: it
rides the existing ``BuiltinToolType`` union, the
``registry.register(..., transport="agent_tool")`` path, the strange-loop
handler template, and the existing skill CRUD service.

The DB is the single source of truth: routing the only sanctioned in-loop edit
through ``create_skill_version`` makes the one-way ``provision_skill_files``
per-step overwrite (``harness/skills.py``) *correct* rather than a clobber — an
attached ``version=None`` skill re-resolves to the latest version next step.

**Identity is load-bearing, so two invariants hold (see F1/F2 in the issue):**
1. The trusted ids (``account_id``, ``session_id``) are NEVER tool-schema fields —
   every arg model is ``extra="forbid"``, so an injected key is rejected before the
   handler runs.
2. Handlers map service kwargs explicitly from the validated model — never
   ``**arguments`` — and ``account_id`` is loaded server-side from the session id.

Both register ``transport="agent_tool"`` (model-only; the CLI broker refuses them).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.services import skills as skills_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import registry

# ─── argument models (parameters_schema + parse, in one place) ───────────────


class _SkillUpsertArgs(BaseModel):
    """``skill_upsert`` arguments.

    ``extra="forbid"`` keeps trusted ids (``account_id``/``session_id``) from ever
    being schema fields — an injected key is rejected before the handler runs (F1).
    ``skill_id is None`` discriminates create-vs-version-add (a KIND, not a flag).
    """

    model_config = ConfigDict(extra="forbid")

    skill_id: str | None = None  # None → create; set → new version of existing
    files: dict[str, str]  # must include exactly one {dir}/SKILL.md
    display_title: str | None = None  # required on create; ignored on version-add


class _SkillArchiveArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str


# ─── handler plumbing ────────────────────────────────────────────────────────
#
# Handlers map service kwargs explicitly (F1) and otherwise let service errors
# propagate: the dispatch layer (``tool_dispatch._classify_tool_error``) turns a
# client-class (4xx) ``AiosError`` — a bad/missing SKILL.md (``ValidationError``,
# 422), an unknown ``skill_id`` (``NotFoundError``, 404) — into a clean,
# model-visible result without evicting the sandbox, and a 5xx into a genuine
# failure. Only argument parsing bails locally (``_parse``).


def _parse[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    """Parse + validate via the pydantic arg model → ``ToolBail`` on failure."""
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc


async def skill_upsert_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_SkillUpsertArgs, arguments)
    if args.skill_id is None:
        if not args.display_title:
            raise ToolBail("display_title required when creating a skill")
        skill, sv = await skills_service.create_skill(
            pool,
            account_id=account_id,
            display_title=args.display_title,
            files=args.files,
        )
        return {
            "skill_id": skill.id,
            "version": sv.version,
            "name": sv.name,
            "directory": sv.directory,
            "created": True,
        }
    sv = await skills_service.create_skill_version(
        pool,
        args.skill_id,
        account_id=account_id,
        files=args.files,
    )
    return {
        "skill_id": args.skill_id,
        "version": sv.version,
        "name": sv.name,
        "directory": sv.directory,
        "created": False,
    }


async def skill_archive_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_SkillArchiveArgs, arguments)
    await skills_service.archive_skill(pool, args.skill_id, account_id=account_id)
    return {"skill_id": args.skill_id, "archived": True}


# ─── descriptions + registration ─────────────────────────────────────────────

SKILL_UPSERT_DESCRIPTION = (
    "Create a new skill or add a new version of one of your own. `files` must "
    "include exactly one `{dir}/SKILL.md` with YAML frontmatter (`name:` "
    "required). Pass `display_title` when creating (omit `skill_id`); to revise "
    "an existing skill pass its `skill_id` (display_title is ignored). Returns "
    "`{skill_id, version, name, directory, created}`."
)
SKILL_ARCHIVE_DESCRIPTION = (
    "Archive one of your skills by id; it disappears from your skill list and "
    "stops projecting into sessions, but its version history is retained."
)


def _register() -> None:
    registry.register(
        name="skill_upsert",
        description=SKILL_UPSERT_DESCRIPTION,
        parameters_schema=_SkillUpsertArgs.model_json_schema(),
        handler=skill_upsert_handler,
        transport="agent_tool",
    )
    registry.register(
        name="skill_archive",
        description=SKILL_ARCHIVE_DESCRIPTION,
        parameters_schema=_SkillArchiveArgs.model_json_schema(),
        handler=skill_archive_handler,
        transport="agent_tool",
    )


_register()
