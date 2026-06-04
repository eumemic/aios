"""``aios skills ...`` — CRUD with directory-walk convenience."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import call_single, render_paginated, unwrap
from aios.cli.coverage import covers
from aios.cli.files import load_payload, walk_skill_dir
from aios.cli.output import print_error, print_success
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.skills import (
    archive_skill,
    create_skill,
    create_skill_version,
    get_skill,
    get_skill_version,
    list_skill_versions,
    list_skills,
)
from aios_sdk._generated.models.skill_create import SkillCreate
from aios_sdk._generated.models.skill_version_create import SkillVersionCreate

app = typer.Typer(name="skills", help="Manage skills.", no_args_is_help=True)

_COLS = ("id", "display_title", "latest_version", "updated_at")


@app.command("list", help="List skills.")
@covers("list_skills")
def list_(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_skills.sync_detailed,
            columns=_COLS,
            all_=all_,
            limit=limit,
        )

    run_or_die(_run)


@app.command("get", help="Fetch a skill.")
@covers("get_skill")
def get(ctx: typer.Context, skill_id: str) -> None:
    def _run() -> None:
        call_single(ctx, get_skill.sync_detailed, skill_id=skill_id)

    run_or_die(_run)


@app.command("create", help="Create a skill from a SKILL.md-rooted directory or a JSON payload.")
@covers("create_skill")
def create(
    ctx: typer.Context,
    dir_: Annotated[
        Path | None,
        typer.Option("--dir", help="Directory containing SKILL.md; walked recursively."),
    ] = None,
    display_title: Annotated[
        str | None,
        typer.Option("--title", help="display_title (required when using --dir)."),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        payload = _build_skill_payload(
            dir_=dir_, title=display_title, file=file, stdin=stdin, data=data
        )
        if isinstance(payload, int):
            return payload
        body = SkillCreate.from_dict(payload)
        call_single(ctx, create_skill.sync_detailed, body=body)
        return None

    run_or_die(_run)


@app.command("archive", help="Archive a skill (soft-delete, retained for audit).")
@covers("archive_skill")
def archive(ctx: typer.Context, skill_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            unwrap(archive_skill.sync_detailed(client=client, skill_id=skill_id))
        print_success("archived", skill_id)

    run_or_die(_run)


@app.command("versions", help="List a skill's version history.")
@covers("list_skill_versions")
def versions(
    ctx: typer.Context,
    skill_id: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_skill_versions.sync_detailed,
            columns=("version", "name", "description", "created_at"),
            max_widths={"description": 60, "name": 32},
            all_=all_,
            limit=limit,
            skill_id=skill_id,
        )

    run_or_die(_run)


@app.command("version", help="Fetch a specific skill version.")
@covers("get_skill_version")
def version(ctx: typer.Context, skill_id: str, version: int) -> None:
    def _run() -> None:
        call_single(ctx, get_skill_version.sync_detailed, skill_id=skill_id, version=version)

    run_or_die(_run)


@app.command("version-create", help="Create a new version of an existing skill.")
@covers("create_skill_version")
def version_create(
    ctx: typer.Context,
    skill_id: str,
    dir_: Annotated[Path | None, typer.Option("--dir")] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        if dir_ is not None:
            files = walk_skill_dir(dir_)
            payload: dict[str, Any] = {"files": files}
        else:
            payload = load_payload(file, stdin, data)
        body = SkillVersionCreate.from_dict(payload)
        call_single(ctx, create_skill_version.sync_detailed, skill_id=skill_id, body=body)
        return None

    run_or_die(_run)


def _build_skill_payload(
    *,
    dir_: Path | None,
    title: str | None,
    file: Path | None,
    stdin: bool,
    data: str | None,
) -> dict[str, Any] | int:
    """Return the SkillCreate payload or an exit code for usage errors."""
    if dir_ is not None:
        if title is None:
            print_error("--title is required when creating a skill from --dir")
            return 64
        files = walk_skill_dir(dir_)
        return {"display_title": title, "files": files}
    return load_payload(file, stdin, data)
