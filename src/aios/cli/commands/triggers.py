"""``aios triggers ...`` — operator-owned triggers (``/v1/triggers``, #2473).

A trigger an operator owns has no session: it fires on a ``cron`` or ``one_shot``
source and launches an operator workflow run with a required ``budget_usd``. A
session's own triggers are ``aios sessions triggers``. Bodies pass through
untyped, so the server is the only validator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.commands._shared import get_state_and_client, raw_single, render_list
from aios.cli.coverage import covers
from aios.cli.files import load_payload
from aios.cli.output import print_error, print_success
from aios.cli.runtime import get_state, run_or_die
from aios_sdk import raw_request

app = typer.Typer(
    name="triggers",
    help="Manage operator-owned triggers (timer-fired operator workflow runs).",
    no_args_is_help=True,
)

_COLS = (
    "id",
    "name",
    "enabled",
    "next_fire",
    "last_fire_status",
    "consecutive_failures",
    "environment_id",
)
_RUN_COLS = (
    "id",
    "trigger_context",
    "status",
    "result_id",
    "error_summary",
    "created_at",
    "finished_at",
)


@app.command("list", help="List the account's operator triggers.")
@covers("list_operator_triggers")
def list_(ctx: typer.Context) -> None:
    def _run() -> None:
        state, client = get_state_and_client(ctx)
        with client:
            envelope = raw_request(client, "GET", "/v1/triggers")
        render_list(state.output_format, envelope, columns=_COLS)

    run_or_die(_run)


@app.command("get", help="Get an operator trigger by name.")
@covers("get_operator_trigger")
def get(ctx: typer.Context, name: str) -> None:
    run_or_die(lambda: raw_single(ctx, "GET", f"/v1/triggers/{name}"))


@app.command("create", help="Create an operator trigger (OperatorTriggerCreate shape).")
@covers("create_operator_trigger")
def create(
    ctx: typer.Context,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> None:
        raw_single(ctx, "POST", "/v1/triggers", json_body=load_payload(file, stdin, data))

    run_or_die(_run)


@app.command("update", help="Update an operator trigger by name (OperatorTriggerUpdate shape).")
@covers("update_operator_trigger")
def update(
    ctx: typer.Context,
    name: str,
    enabled: Annotated[
        bool | None, typer.Option("--enabled/--disabled", show_default=False)
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        if any([file, stdin, data]):
            payload = load_payload(file, stdin, data)
        elif enabled is not None:
            payload = {"enabled": enabled}
        else:
            print_error(
                "provide --enabled/--disabled, or a payload via --file/--stdin/--data "
                "(source/action replace the stored object wholesale)."
            )
            return 64
        raw_single(ctx, "PUT", f"/v1/triggers/{name}", json_body=payload)
        return None

    run_or_die(_run)


@app.command("delete", help="Delete an operator trigger by name.")
@covers("delete_operator_trigger")
def delete(ctx: typer.Context, name: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            raw_request(client, "DELETE", f"/v1/triggers/{name}")
        print_success("deleted", name)

    run_or_die(_run)


@app.command("runs", help="List an operator trigger's fires, newest first.")
@covers("list_operator_trigger_runs")
def runs(
    ctx: typer.Context,
    name: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
) -> None:
    def _run() -> None:
        state, client = get_state_and_client(ctx)
        with client:
            envelope = raw_request(
                client, "GET", f"/v1/triggers/{name}/runs", params={"limit": limit}
            )
        render_list(state.output_format, envelope, columns=_RUN_COLS)

    run_or_die(_run)
