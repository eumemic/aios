"""``aios envs ...`` — environment (sandbox config) CRUD."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.commands._shared import call_single, render_paginated, unwrap
from aios.cli.coverage import covers
from aios.cli.files import load_payload
from aios.cli.output import print_success
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.environments import (
    archive_environment,
    create_environment,
    get_environment,
    list_environments,
    update_environment,
)
from aios_sdk._generated.models.environment_create import EnvironmentCreate
from aios_sdk._generated.models.environment_update import EnvironmentUpdate

app = typer.Typer(name="envs", help="Manage environments (sandbox configs).", no_args_is_help=True)

_COLS = ("id", "name", "archived_at", "updated_at")


@app.command("list")
@covers("list_environments")
def list_(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_environments.sync_detailed,
            columns=_COLS,
            all_=all_,
            limit=limit,
        )

    run_or_die(_run)


@app.command("get")
@covers("get_environment")
def get(ctx: typer.Context, env_id: str) -> None:
    def _run() -> None:
        call_single(ctx, get_environment.sync_detailed, env_id=env_id)

    run_or_die(_run)


@app.command("create", help="Create an environment (EnvironmentCreate shape).")
@covers("create_environment")
def create(
    ctx: typer.Context,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        payload = load_payload(file, stdin, data)
        body = EnvironmentCreate.from_dict(payload)
        call_single(ctx, create_environment.sync_detailed, body=body)
        return None

    run_or_die(_run)


@app.command("update", help="Update an environment (EnvironmentUpdate shape).")
@covers("update_environment")
def update(
    ctx: typer.Context,
    env_id: str,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        payload = load_payload(file, stdin, data)
        body = EnvironmentUpdate.from_dict(payload)
        call_single(ctx, update_environment.sync_detailed, env_id=env_id, body=body)
        return None

    run_or_die(_run)


@app.command("archive", help="Archive an environment (soft-delete, retained for audit).")
@covers("archive_environment")
def archive(ctx: typer.Context, env_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            unwrap(archive_environment.sync_detailed(client=client, env_id=env_id))
        print_success("archived", env_id)

    run_or_die(_run)
