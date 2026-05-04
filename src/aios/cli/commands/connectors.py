"""``aios connectors ...`` — connector subprocess admin.

Thin client over ``GET /v1/connectors`` and friends.  All commands
flow through the API process, which procrastinate-RPCs into the
worker (see :mod:`aios.api.routers.connectors`).  The ``call``
subcommand expects a JSON payload of shape ``{"tool": str,
"arguments": dict, "meta": dict | null}``.

Multi-instance: most commands take ``<connector> [<instance>]``
positional args.  When the operator omits ``<instance>`` and the
connector type has exactly one configured instance, that instance is
used; if multiple are configured the CLI errors out with a list and
the operator must specify.  This sugar lives in the CLI only — the
API + procrastinate task layer always require explicit pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

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


def _resolve_instance(ctx: typer.Context, connector: str, instance: str | None) -> str:
    """Auto-resolve instance when only one of this connector type is configured.

    Lists the connector's instances via ``GET /v1/connectors/<connector>``.
    If exactly one, return its instance name.  If multiple, raise typer
    Exit with a list so the operator picks explicitly.  If none, the
    enclosing API call will surface the not-enabled error itself.
    """
    if instance is not None:
        return instance
    client = just_client(ctx)
    with client:
        obj = client.request("GET", f"/v1/connectors/{connector}")
    instances: list[dict[str, Any]] = obj.get("connectors", []) if isinstance(obj, dict) else []
    names = [entry.get("instance") for entry in instances if isinstance(entry, dict)]
    if len(names) == 1 and isinstance(names[0], str):
        return names[0]
    if not names:
        raise typer.Exit(code=1)
    print_error(
        f"connector {connector!r} has multiple instances ({', '.join(str(n) for n in names)}); "
        "specify one explicitly"
    )
    raise typer.Exit(code=2)


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
        resolved = _resolve_instance(ctx, connector, instance)
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/connectors/{connector}/{resolved}/accounts")
        render_single(obj)

    run_or_die(_run)


@app.command("tools")
def tools(
    ctx: typer.Context,
    connector: str,
    instance: Annotated[str | None, typer.Argument()] = None,
) -> None:
    def _run() -> None:
        resolved = _resolve_instance(ctx, connector, instance)
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/connectors/{connector}/{resolved}/tools")
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
        resolved = _resolve_instance(ctx, connector, instance)
        client = just_client(ctx)
        with client:
            obj = client.request(
                "POST", f"/v1/connectors/{connector}/{resolved}/call", json_body=payload
            )
        render_single(obj)
        return None

    run_or_die(_run)
