"""``aios rules ...`` — routing-rules CRUD, scoped to a connection."""

from __future__ import annotations

import json
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


@app.command("create", help="Create a routing rule.")
def create(
    ctx: typer.Context,
    connection_id: str,
    prefix: Annotated[
        str | None,
        typer.Option("--prefix", help='Channel-path prefix ("" = per-connection catch-all).'),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            help='Route target, e.g. "agent:claude-sonnet-4" or "session:sess_01".',
        ),
    ] = None,
    session_params_json: Annotated[
        str | None,
        typer.Option(
            "--session-params-json",
            help="JSON object for nested SessionParams (default: empty object).",
        ),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        ergonomic = prefix is not None or target is not None
        if ergonomic:
            if any([file, stdin, data]):
                print_error("combine ergonomic flags OR --file/--stdin/--data, not both")
                return 64
            if prefix is None or target is None:
                print_error("--prefix and --target are both required")
                return 64
            payload: dict[str, Any] = {"prefix": prefix, "target": target}
            if session_params_json is not None:
                try:
                    payload["session_params"] = json.loads(session_params_json)
                except json.JSONDecodeError as exc:
                    print_error(f"invalid --session-params-json: {exc}")
                    return 64
            else:
                payload["session_params"] = {}
        else:
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


@app.command("archive", help="Archive a routing rule (soft-delete, retained for audit).")
def archive(ctx: typer.Context, connection_id: str, rule_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/connections/{connection_id}/routing-rules/{rule_id}")

    run_or_die(_run)
