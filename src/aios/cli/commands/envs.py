"""``aios envs ...`` — environment (sandbox config) CRUD."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.commands._shared import (
    fetch_all,
    just_client,
    render_list,
    render_single,
    with_client,
)
from aios.cli.files import PayloadError, load_payload
from aios.cli.output import print_error
from aios.cli.runtime import run_or_die

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
        state, client = with_client(ctx)
        with client:
            envelope = (
                fetch_all(client, "/v1/environments")
                if all_
                else client.request(
                    "GET", "/v1/environments", params={"limit": limit, "after": after}
                )
            )
        render_list(state.output_format, envelope, columns=_COLS)

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, env_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/environments/{env_id}")
        render_single(obj)

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
        client = just_client(ctx)
        with client:
            obj = client.request("POST", "/v1/environments", json_body=payload)
        render_single(obj)
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
        client = just_client(ctx)
        with client:
            obj = client.request("PUT", f"/v1/environments/{env_id}", json_body=payload)
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("delete")
def delete(ctx: typer.Context, env_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/environments/{env_id}")

    run_or_die(_run)
