"""``aios skills ...`` — CRUD with directory-walk convenience."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import (
    fetch_all,
    just_client,
    render_list,
    render_single,
    with_client,
)
from aios.cli.files import PayloadError, load_payload, walk_skill_dir
from aios.cli.output import print_error, print_success
from aios.cli.runtime import run_or_die

app = typer.Typer(name="skills", help="Manage skills.", no_args_is_help=True)

_COLS = ("id", "display_title", "latest_version", "updated_at")


@app.command("list", help="List skills.")
def list_(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        with client:
            envelope = (
                fetch_all(client, "/v1/skills")
                if all_
                else client.request("GET", "/v1/skills", params={"limit": limit, "after": after})
            )
        render_list(state.output_format, envelope, columns=_COLS)

    run_or_die(_run)


@app.command("get", help="Fetch a skill.")
def get(ctx: typer.Context, skill_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/skills/{skill_id}")
        render_single(obj)

    run_or_die(_run)


@app.command("create", help="Create a skill from a SKILL.md-rooted directory or a JSON payload.")
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
        client = just_client(ctx)
        with client:
            obj = client.request("POST", "/v1/skills", json_body=payload)
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("archive", help="Archive a skill (soft-delete, retained for audit).")
def archive(ctx: typer.Context, skill_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/skills/{skill_id}")
        print_success("archived", skill_id)

    run_or_die(_run)


@app.command("versions", help="List a skill's version history.")
def versions(
    ctx: typer.Context,
    skill_id: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        path = f"/v1/skills/{skill_id}/versions"
        with client:
            envelope = (
                fetch_all(client, path)
                if all_
                else client.request("GET", path, params={"limit": limit, "after": after})
            )
        render_list(
            state.output_format,
            envelope,
            columns=("version", "name", "description", "created_at"),
            max_widths={"description": 60, "name": 32},
        )

    run_or_die(_run)


@app.command("version", help="Fetch a specific skill version.")
def version(ctx: typer.Context, skill_id: str, version: int) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/skills/{skill_id}/versions/{version}")
        render_single(obj)

    run_or_die(_run)


@app.command("version-create", help="Create a new version of an existing skill.")
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
            try:
                files = walk_skill_dir(dir_)
            except PayloadError as exc:
                print_error(str(exc))
                return 64
            payload: dict[str, Any] = {"files": files}
        else:
            try:
                payload = load_payload(file, stdin, data)
            except PayloadError as exc:
                print_error(str(exc))
                return 64
        client = just_client(ctx)
        with client:
            obj = client.request("POST", f"/v1/skills/{skill_id}/versions", json_body=payload)
        render_single(obj)
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
        try:
            files = walk_skill_dir(dir_)
        except PayloadError as exc:
            print_error(str(exc))
            return 64
        return {"display_title": title, "files": files}
    try:
        return load_payload(file, stdin, data)
    except PayloadError as exc:
        print_error(str(exc))
        return 64
