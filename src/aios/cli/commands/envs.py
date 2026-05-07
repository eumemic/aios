"""``aios envs ...`` — environment (sandbox config) CRUD."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.commands._shared import render_paginated, render_single, unwrap
from aios.cli.files import PayloadError, load_payload
from aios.cli.output import print_error, print_success
from aios.cli.runtime import get_state, run_or_die
from aios.sdk._generated.api.environments import (
    archive_environment,
    create_environment,
    get_environment,
    list_environments,
    update_environment,
)
from aios.sdk._generated.models.environment_create import EnvironmentCreate
from aios.sdk._generated.models.environment_update import EnvironmentUpdate

app = typer.Typer(name="envs", help="Manage environments (sandbox configs).", no_args_is_help=True)

_COLS = ("id", "name", "archived_at", "updated_at")


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
            list_environments.sync_detailed,
            columns=_COLS,
            all_=all_,
            limit=limit,
            after=after,
        )

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, env_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(get_environment.sync_detailed(client=client, env_id=env_id))
        render_single(obj.to_dict())

    run_or_die(_run)


@app.command("create", help="Create an environment (EnvironmentCreate shape).")
def create(
    ctx: typer.Context,
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
        body = EnvironmentCreate.from_dict(payload)
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(create_environment.sync_detailed(client=client, body=body))
        render_single(obj.to_dict())
        return None

    run_or_die(_run)


@app.command("update", help="Update an environment (EnvironmentUpdate shape).")
def update(
    ctx: typer.Context,
    env_id: str,
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
        body = EnvironmentUpdate.from_dict(payload)
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(update_environment.sync_detailed(client=client, env_id=env_id, body=body))
        render_single(obj.to_dict())
        return None

    run_or_die(_run)


@app.command("archive", help="Archive an environment (soft-delete, retained for audit).")
def archive(ctx: typer.Context, env_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            unwrap(archive_environment.sync_detailed(client=client, env_id=env_id))
        print_success("archived", env_id)

    run_or_die(_run)
