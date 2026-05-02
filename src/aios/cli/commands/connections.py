"""``aios connections ...`` — connector-instance CRUD plus mode-binding helpers.

The connector redesign (#200) unified routing into the ``connections``
table.  Connections are created in detached mode; switch to single_session
via ``attach`` or per_chat via ``configure-per-chat``.  ``archive`` works
only on detached connections — operators must ``detach`` /
``unconfigure`` first.
"""

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
from aios.cli.files import PayloadError, load_json_object, resolve_payload
from aios.cli.output import print_error, print_success
from aios.cli.runtime import run_or_die

app = typer.Typer(name="connections", help="Manage connector connections.", no_args_is_help=True)

_COLS = ("id", "connector", "account", "session_id", "session_template_id", "updated_at")
_MAXW = {"connector": 20, "account": 40, "session_id": 24, "session_template_id": 24}


@app.command("list")
def list_(
    ctx: typer.Context,
    connector: Annotated[str | None, typer.Option("--connector")] = None,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        params: dict[str, Any] = {"connector": connector, "session_id": session_id}
        with client:
            envelope = (
                fetch_all(client, "/v1/connections", params=params)
                if all_
                else client.request(
                    "GET",
                    "/v1/connections",
                    params={**params, "limit": limit, "after": after},
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


@app.command("create", help="Create a connection in detached mode.")
def create(
    ctx: typer.Context,
    connector: Annotated[
        str | None, typer.Option("--connector", help="Connector type (e.g. signal).")
    ] = None,
    account: Annotated[
        str | None, typer.Option("--account", help="Account identifier (e.g. bot uuid).")
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
        if connector is not None or account is not None:
            if connector is None or account is None:
                print_error("--connector and --account are both required")
                return 64
            ergonomic = {"connector": connector, "account": account}
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


@app.command("attach", help="Bind a detached connection to a session (single_session mode).")
def attach(
    ctx: typer.Context,
    connection_id: str,
    session_id: Annotated[str, typer.Option("--session-id")],
) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request(
                "POST",
                f"/v1/connections/{connection_id}/attach",
                json_body={"session_id": session_id},
            )
        render_single(obj)

    run_or_die(_run)


@app.command("detach", help="Drop the single_session binding, leaving the connection detached.")
def detach(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("POST", f"/v1/connections/{connection_id}/detach")
        render_single(obj)

    run_or_die(_run)


@app.command("configure-per-chat", help="Switch the connection into per_chat mode.")
def configure_per_chat(
    ctx: typer.Context,
    connection_id: str,
    template: Annotated[str, typer.Option("--template", help="Session template id.")],
) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request(
                "POST",
                f"/v1/connections/{connection_id}/configure-per-chat",
                json_body={"session_template_id": template},
            )
        render_single(obj)

    run_or_die(_run)


@app.command(
    "unconfigure", help="Drop the per_chat configuration, leaving the connection detached."
)
def unconfigure(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("POST", f"/v1/connections/{connection_id}/unconfigure")
        render_single(obj)

    run_or_die(_run)


@app.command("archive", help="Archive a detached connection (soft-delete, retained for audit).")
def archive(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/connections/{connection_id}")
        print_success("archived", connection_id)

    run_or_die(_run)
