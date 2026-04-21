"""``aios agents ...`` — CRUD + versions."""

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
        state, client = with_client(ctx)
        with client:
            if all_:
                envelope = fetch_all(client, "/v1/agents")
            else:
                envelope = client.request(
                    "GET", "/v1/agents", params={"limit": limit, "after": after}
                )
        render_list(state.output_format, envelope, columns=_COLS, max_widths=_MAXW)

    run_or_die(_run)


@app.command("get", help="Fetch a single agent by id.")
def get(ctx: typer.Context, agent_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            agent = client.request("GET", f"/v1/agents/{agent_id}")
        render_single(agent)

    run_or_die(_run)


@app.command("create", help="Create an agent from a JSON payload (AgentCreate shape).")
def create(
    ctx: typer.Context,
    file: Annotated[Path | None, typer.Option("--file", help="Read JSON body from a file.")] = None,
    stdin: Annotated[bool, typer.Option("--stdin", help="Read JSON body from stdin.")] = False,
    data: Annotated[str | None, typer.Option("--data", help="Inline JSON body.")] = None,
) -> None:
    def _run() -> int | None:
        try:
            payload = load_payload(file, stdin, data)
        except PayloadError as exc:
            print_error(str(exc))
            return 64
        client = just_client(ctx)
        with client:
            agent = client.request("POST", "/v1/agents", json_body=payload)
        render_single(agent)
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
        try:
            payload = load_payload(file, stdin, data)
        except PayloadError as exc:
            print_error(str(exc))
            return 64
        client = just_client(ctx)
        with client:
            agent = client.request("PUT", f"/v1/agents/{agent_id}", json_body=payload)
        render_single(agent)
        return None

    run_or_die(_run)


@app.command("delete", help="Soft-archive an agent.")
def delete(ctx: typer.Context, agent_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/agents/{agent_id}")

    run_or_die(_run)


@app.command("versions", help="List an agent's version history.")
def versions(
    ctx: typer.Context,
    agent_id: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        with client:
            if all_:
                envelope = fetch_all(client, f"/v1/agents/{agent_id}/versions")
            else:
                envelope = client.request(
                    "GET",
                    f"/v1/agents/{agent_id}/versions",
                    params={"limit": limit, "after": after},
                )
        render_list(
            state.output_format,
            envelope,
            columns=("version", "model", "created_at"),
            max_widths={"model": 40},
        )

    run_or_die(_run)


@app.command("version", help="Fetch a specific agent version.")
def version(ctx: typer.Context, agent_id: str, version: int) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/agents/{agent_id}/versions/{version}")
        render_single(obj)

    run_or_die(_run)
