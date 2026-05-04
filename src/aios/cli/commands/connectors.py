"""``aios connectors ...`` — connector subprocess admin.

Thin client over ``GET /v1/connectors`` and friends.  All commands
flow through the API process, which procrastinate-RPCs into the
worker (see :mod:`aios.api.routers.connectors`).  The ``call``
subcommand expects a JSON payload of shape ``{"tool": str,
"arguments": dict, "meta": dict | null}``.

Multi-instance: most commands take ``<connector> [<instance>]``
positional args.  When the operator omits ``<instance>``, the CLI
passes the ``_`` sentinel to the API; the worker auto-resolves it to
the sole enabled instance, or returns 409 with the list if multiple
are configured.  One round-trip either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.commands._shared import just_client, render_single
from aios.cli.files import PayloadError, load_payload
from aios.cli.output import print_error
from aios.cli.runtime import run_or_die
from aios.config import DEFAULT_INSTANCE_SENTINEL

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


@app.command("get")
def get(
    ctx: typer.Context,
    connector: str,
    instance: Annotated[str | None, typer.Argument()] = None,
) -> None:
    """Show all instances of a connector type, or one specific instance."""

    def _run() -> None:
        client = just_client(ctx)
        path = f"/v1/connectors/{connector}"
        if instance is not None:
            path = f"{path}/{instance}"
        with client:
            obj = client.request("GET", path)
        render_single(obj)

    run_or_die(_run)


@app.command("accounts")
def accounts(
    ctx: typer.Context,
    connector: str,
    instance: Annotated[str | None, typer.Argument()] = None,
) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request(
                "GET",
                f"/v1/connectors/{connector}/{instance or DEFAULT_INSTANCE_SENTINEL}/accounts",
            )
        render_single(obj)

    run_or_die(_run)


@app.command("tools")
def tools(
    ctx: typer.Context,
    connector: str,
    instance: Annotated[str | None, typer.Argument()] = None,
) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request(
                "GET", f"/v1/connectors/{connector}/{instance or DEFAULT_INSTANCE_SENTINEL}/tools"
            )
        render_single(obj)

    run_or_die(_run)


@app.command("call", help="Invoke a connector tool by name (admin escape hatch).")
def call(
    ctx: typer.Context,
    connector: str,
    instance: Annotated[str | None, typer.Argument()] = None,
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
            obj = client.request(
                "POST",
                f"/v1/connectors/{connector}/{instance or DEFAULT_INSTANCE_SENTINEL}/call",
                json_body=payload,
            )
        render_single(obj)
        return None

    run_or_die(_run)
