"""``aios connectors ...`` — connector subprocess admin.

The connector subprocess supervisor isn't wired yet; every subcommand
passes through to the API, which currently returns 503.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.commands._shared import just_client, render_single
from aios.cli.files import PayloadError, load_payload
from aios.cli.output import print_error
from aios.cli.runtime import run_or_die

app = typer.Typer(
    name="connectors",
    help="Inspect connector subprocesses (admin).",
    no_args_is_help=True,
)


@app.command("list")
def list_(ctx: typer.Context) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", "/v1/connectors")
        render_single(obj)

    run_or_die(_run)


@app.command("accounts")
def accounts(ctx: typer.Context, name: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/connectors/{name}/accounts")
        render_single(obj)

    run_or_die(_run)


@app.command("tools")
def tools(ctx: typer.Context, name: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/connectors/{name}/tools")
        render_single(obj)

    run_or_die(_run)


@app.command("call", help="Invoke a connector tool by name (admin escape hatch).")
def call(
    ctx: typer.Context,
    name: str,
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
            obj = client.request("POST", f"/v1/connectors/{name}/call", json_body=payload)
        render_single(obj)
        return None

    run_or_die(_run)
