"""``aios invocations ...`` — the API caller's request-writer surface (#1128).

One kind-agnostic command that writes a trusted request edge + resolves-or-creates
a servicer (a session or a run) and prints the structured handle
(``servicer_kind``, ``servicer_id``, ``request_id``). The handle is the
ephemeral caller's continuation: await it with ``aios sessions await`` (for
``servicer_kind=session``) or ``aios runs wait`` (for ``servicer_kind=run``).
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import call_single
from aios.cli.coverage import covers
from aios.cli.files import PayloadError
from aios.cli.runtime import run_or_die
from aios_sdk._generated.api.invocations import invoke
from aios_sdk._generated.models.invocation_request import InvocationRequest

app = typer.Typer(
    name="invocations",
    help="Write a request edge + resolve-or-create a servicer (the API caller surface).",
    no_args_is_help=True,
)


def _json_arg(value: str | None) -> Any:
    """Parse a JSON CLI argument (``--input`` / ``--output-schema``); ``None`` stays ``None``."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise PayloadError(f"invalid JSON: {exc}") from exc


@app.command("create", help="Invoke a target and print the durable handle.")
@covers("invoke")
def create_(
    ctx: typer.Context,
    target_kind: Annotated[
        str, typer.Option("--target-kind", help="One of: agent | workflow | session.")
    ],
    target: Annotated[
        str, typer.Option("--target", help="An agent_id / workflow_id / session_id.")
    ],
    input_: Annotated[
        str | None, typer.Option("--input", help="The request payload as JSON.")
    ] = None,
    output_schema: Annotated[
        str | None,
        typer.Option("--output-schema", help="JSON Schema the response value must satisfy."),
    ] = None,
    environment_id: Annotated[
        str | None,
        typer.Option(
            "--environment-id",
            help="Environment to bind a created servicer to (agent/workflow only).",
        ),
    ] = None,
) -> None:
    def _run() -> int | None:
        body = InvocationRequest.from_dict(
            {
                "target_kind": target_kind,
                "target": target,
                "input": _json_arg(input_),
                "output_schema": _json_arg(output_schema),
                "environment_id": environment_id,
            }
        )
        call_single(ctx, invoke.sync_detailed, body=body)
        return None

    run_or_die(_run)
