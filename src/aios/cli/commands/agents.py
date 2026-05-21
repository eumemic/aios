"""``aios agents ...`` — CRUD + versions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.commands._shared import call_single, render_paginated, unwrap
from aios.cli.files import load_payload
from aios.cli.output import print_success
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.agents import (
    archive_agent,
    create_agent,
    get_agent,
    get_agent_version,
    list_agent_versions,
    list_agents,
    update_agent,
)
from aios_sdk._generated.models.agent_create import AgentCreate
from aios_sdk._generated.models.agent_update import AgentUpdate

app = typer.Typer(name="agents", help="Manage agents.", no_args_is_help=True)

_COLS = ("id", "name", "model", "updated_at")
_MAXW = {"name": 32, "model": 40}


@app.command("list", help="List agents.")
def list_(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after", help="Cursor for pagination.")] = None,
    all_: Annotated[
        bool, typer.Option("--all", help="Fetch every page (ignores --limit/--after).")
    ] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_agents.sync_detailed,
            columns=_COLS,
            max_widths=_MAXW,
            all_=all_,
            limit=limit,
            after=after,
        )

    run_or_die(_run)


@app.command("get", help="Fetch a single agent by id.")
def get(ctx: typer.Context, agent_id: str) -> None:
    def _run() -> None:
        call_single(ctx, get_agent.sync_detailed, agent_id=agent_id)

    run_or_die(_run)


@app.command("create", help="Create an agent from a JSON payload (AgentCreate shape).")
def create(
    ctx: typer.Context,
    file: Annotated[Path | None, typer.Option("--file", help="Read JSON body from a file.")] = None,
    stdin: Annotated[bool, typer.Option("--stdin", help="Read JSON body from stdin.")] = False,
    data: Annotated[str | None, typer.Option("--data", help="Inline JSON body.")] = None,
) -> None:
    def _run() -> int | None:
        payload = load_payload(file, stdin, data)
        body = AgentCreate.from_dict(payload)
        call_single(ctx, create_agent.sync_detailed, body=body)
        return None

    run_or_die(_run)


@app.command("update", help="Update an agent (AgentUpdate shape; include 'version').")
def update(
    ctx: typer.Context,
    agent_id: str,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        payload = load_payload(file, stdin, data)
        body = AgentUpdate.from_dict(payload)
        call_single(ctx, update_agent.sync_detailed, agent_id=agent_id, body=body)
        return None

    run_or_die(_run)


@app.command("archive", help="Archive an agent (soft-delete, retained for audit).")
def archive(ctx: typer.Context, agent_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            unwrap(archive_agent.sync_detailed(client=client, agent_id=agent_id))
        print_success("archived", agent_id)

    run_or_die(_run)


@app.command("versions", help="List an agent's version history.")
def versions(
    ctx: typer.Context,
    agent_id: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[int | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_agent_versions.sync_detailed,
            columns=("version", "model", "created_at"),
            max_widths={"model": 40},
            all_=all_,
            limit=limit,
            after=after,
            agent_id=agent_id,
        )

    run_or_die(_run)


@app.command("version", help="Fetch a specific agent version.")
def version(ctx: typer.Context, agent_id: str, version: int) -> None:
    def _run() -> None:
        call_single(ctx, get_agent_version.sync_detailed, agent_id=agent_id, version=version)

    run_or_die(_run)
