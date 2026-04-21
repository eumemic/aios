"""``aios connections ...`` — connector-instance CRUD + inbound-message helper."""

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
from aios.cli.files import PayloadError, load_json_object, load_payload, resolve_payload
from aios.cli.output import print_error
from aios.cli.runtime import run_or_die

app = typer.Typer(name="connections", help="Manage connector connections.", no_args_is_help=True)

_COLS = ("id", "connector", "account", "mcp_url", "updated_at")
_MAXW = {"connector": 20, "account": 40, "mcp_url": 40}


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
                fetch_all(client, "/v1/connections")
                if all_
                else client.request(
                    "GET", "/v1/connections", params={"limit": limit, "after": after}
                )
            )
        render_list(state.output_format, envelope, columns=_COLS, max_widths=_MAXW)

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/connections/{connection_id}")
        render_single(obj)

    run_or_die(_run)


@app.command("create", help="Create a connection.")
def create(
    ctx: typer.Context,
    connector: Annotated[
        str | None, typer.Option("--connector", help="Connector type (e.g. signal).")
    ] = None,
    account: Annotated[
        str | None, typer.Option("--account", help="Account identifier (e.g. bot uuid).")
    ] = None,
    mcp_url: Annotated[str | None, typer.Option("--mcp-url", help="MCP server URL.")] = None,
    vault_id: Annotated[
        str | None, typer.Option("--vault-id", help="Vault id with the MCP credential.")
    ] = None,
    metadata_json: Annotated[
        str | None,
        typer.Option("--metadata-json", help="JSON object of connection metadata."),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        ergonomic: dict[str, Any] | None = None
        if any(v is not None for v in (connector, account, mcp_url, vault_id)):
            missing = [
                name
                for name, v in (
                    ("--connector", connector),
                    ("--account", account),
                    ("--mcp-url", mcp_url),
                    ("--vault-id", vault_id),
                )
                if v is None
            ]
            if missing:
                print_error(f"missing required flag(s): {', '.join(missing)}")
                return 64
            ergonomic = {
                "connector": connector,
                "account": account,
                "mcp_url": mcp_url,
                "vault_id": vault_id,
            }
            if metadata_json is not None:
                try:
                    ergonomic["metadata"] = load_json_object(metadata_json, "--metadata-json")
                except PayloadError as exc:
                    print_error(str(exc))
                    return 64
        try:
            payload = resolve_payload(ergonomic, file, stdin, data)
        except PayloadError as exc:
            print_error(str(exc))
            return 64
        client = just_client(ctx)
        with client:
            obj = client.request("POST", "/v1/connections", json_body=payload)
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("update", help="Update a connection (ConnectionUpdate shape).")
def update(
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
            obj = client.request("PUT", f"/v1/connections/{connection_id}", json_body=payload)
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("archive", help="Archive a connection (soft-delete, retained for audit).")
def archive(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/connections/{connection_id}")

    run_or_die(_run)


@app.command(
    "inbound",
    help="Post an InboundMessage to a connection (simulates a connector delivery).",
)
def inbound(
    ctx: typer.Context,
    connection_id: str,
    path: Annotated[str, typer.Option("--path", help="Channel path segment after connection.")],
    content: Annotated[str, typer.Option("--content")],
    metadata: Annotated[
        str | None, typer.Option("--metadata", help="Optional JSON metadata object.")
    ] = None,
) -> None:
    def _run() -> int | None:
        body: dict[str, Any] = {"path": path, "content": content}
        if metadata is not None:
            try:
                body["metadata"] = load_json_object(metadata, "--metadata")
            except PayloadError as exc:
                print_error(str(exc))
                return 64
        client = just_client(ctx)
        with client:
            obj = client.request(
                "POST", f"/v1/connections/{connection_id}/messages", json_body=body
            )
        render_single(obj)
        return None

    run_or_die(_run)
