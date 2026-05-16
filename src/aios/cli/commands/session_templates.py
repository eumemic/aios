"""``aios session-templates ...`` — frozen recipes for per_chat session spawn."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import call_single, render_paginated, unwrap
from aios.cli.files import PayloadError, load_json_object, load_payload, resolve_payload
from aios.cli.output import print_error, print_success
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.session_templates import (
    archive_session_template,
    create_session_template,
    get_session_template,
    list_session_templates,
    update_session_template,
)
from aios_sdk._generated.models.session_template_create import SessionTemplateCreate
from aios_sdk._generated.models.session_template_update import SessionTemplateUpdate

app = typer.Typer(
    name="session-templates",
    help="Manage session templates (per_chat session recipes).",
    no_args_is_help=True,
)

_COLS = ("id", "name", "agent_id", "environment_id", "updated_at")
_MAXW = {"name": 30, "agent_id": 24, "environment_id": 24}


@app.command("list")
def list_(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_session_templates.sync_detailed,
            columns=_COLS,
            max_widths=_MAXW,
            all_=all_,
            limit=limit,
            after=after,
        )

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, template_id: str) -> None:
    def _run() -> None:
        call_single(ctx, get_session_template.sync_detailed, template_id=template_id)

    run_or_die(_run)


@app.command("create", help="Create a session template.")
def create(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Option("--name")] = None,
    agent_id: Annotated[str | None, typer.Option("--agent-id")] = None,
    environment_id: Annotated[str | None, typer.Option("--environment-id")] = None,
    agent_version: Annotated[int | None, typer.Option("--agent-version")] = None,
    metadata_json: Annotated[
        str | None,
        typer.Option("--metadata-json", help="JSON object of template metadata."),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        ergonomic: dict[str, Any] | None = None
        if any(v is not None for v in (name, agent_id, environment_id)):
            missing = [
                flag
                for flag, v in (
                    ("--name", name),
                    ("--agent-id", agent_id),
                    ("--environment-id", environment_id),
                )
                if v is None
            ]
            if missing:
                print_error(f"missing required flag(s): {', '.join(missing)}")
                return 64
            ergonomic = {
                "name": name,
                "agent_id": agent_id,
                "environment_id": environment_id,
            }
            if agent_version is not None:
                ergonomic["agent_version"] = agent_version
            if metadata_json is not None:
                try:
                    ergonomic["metadata"] = load_json_object(metadata_json, "--metadata-json")
                except PayloadError as exc:
                    print_error(str(exc))
                    return 64
        try:
            payload = resolve_payload(ergonomic, file, stdin, data)
        except PayloadError as exc:
            print_error(str(exc))
            return 64
        body = SessionTemplateCreate.from_dict(payload)
        call_single(ctx, create_session_template.sync_detailed, body=body)
        return None

    run_or_die(_run)


@app.command("update", help="Update a session template (SessionTemplateUpdate shape).")
def update(
    ctx: typer.Context,
    template_id: str,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        try:
            payload = load_payload(file, stdin, data)
        except PayloadError as exc:
            print_error(str(exc))
            return 64
        body = SessionTemplateUpdate.from_dict(payload)
        call_single(ctx, update_session_template.sync_detailed, template_id=template_id, body=body)
        return None

    run_or_die(_run)


@app.command("archive", help="Archive a session template (soft-delete, retained for audit).")
def archive(ctx: typer.Context, template_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            unwrap(archive_session_template.sync_detailed(client=client, template_id=template_id))
        print_success("archived", template_id)

    run_or_die(_run)
