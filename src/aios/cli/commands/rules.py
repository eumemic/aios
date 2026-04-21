"""``aios rules ...`` — routing-rules CRUD, scoped to a connection."""

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

app = typer.Typer(
    name="rules",
    help="Manage connection routing rules.",
    no_args_is_help=True,
)

_COLS = ("id", "prefix", "target", "created_at")
_MAXW = {"prefix": 40, "target": 60}


@app.command("list")
def list_(
    ctx: typer.Context,
    connection_id: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        path = f"/v1/connections/{connection_id}/routing-rules"
        with client:
            envelope = (
                fetch_all(client, path)
                if all_
                else client.request("GET", path, params={"limit": limit, "after": after})
            )
        render_list(state.output_format, envelope, columns=_COLS, max_widths=_MAXW)

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, connection_id: str, rule_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/connections/{connection_id}/routing-rules/{rule_id}")
        render_single(obj)

    run_or_die(_run)


@app.command("create", help="Create a routing rule (RoutingRuleCreate shape).")
def create(
    ctx: typer.Context,
    connection_id: str,
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
                "POST", f"/v1/connections/{connection_id}/routing-rules", json_body=payload
            )
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("update", help="Update a routing rule (RoutingRuleUpdate shape).")
def update(
    ctx: typer.Context,
    connection_id: str,
    rule_id: str,
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
                "PUT",
                f"/v1/connections/{connection_id}/routing-rules/{rule_id}",
                json_body=payload,
            )
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("delete")
def delete(ctx: typer.Context, connection_id: str, rule_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/connections/{connection_id}/routing-rules/{rule_id}")

    run_or_die(_run)
