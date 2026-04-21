"""``aios bindings ...`` — channel-binding CRUD."""

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
from aios.cli.files import PayloadError, load_payload
from aios.cli.output import print_error
from aios.cli.runtime import run_or_die

app = typer.Typer(name="bindings", help="Manage channel bindings.", no_args_is_help=True)

_COLS = ("id", "address", "session_id", "created_at")
_MAXW = {"address": 60, "session_id": 32}


@app.command("list")
def list_(
    ctx: typer.Context,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        params: dict[str, Any] = {"session_id": session_id}
        with client:
            if all_:
                envelope = fetch_all(client, "/v1/channel-bindings", params=params)
            else:
                envelope = client.request(
                    "GET",
                    "/v1/channel-bindings",
                    params={**params, "limit": limit, "after": after},
                )
        render_list(state.output_format, envelope, columns=_COLS, max_widths=_MAXW)

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, binding_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/channel-bindings/{binding_id}")
        render_single(obj)

    run_or_die(_run)


@app.command("create", help="Create a channel binding (address + session_id).")
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
            obj = client.request("POST", "/v1/channel-bindings", json_body=payload)
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("delete")
def delete(ctx: typer.Context, binding_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/channel-bindings/{binding_id}")

    run_or_die(_run)
