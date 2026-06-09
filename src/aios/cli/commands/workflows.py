"""``aios workflows ...`` (definitions) + ``aios runs ...`` (executions).

Two typer sub-apps mirroring the two routers. ``runs`` favours convenience flags
(launch + resume are usually ad-hoc) while ``workflows create`` takes a JSON
payload (the script body is long). ``--input`` / ``--result`` are parsed as JSON
(a workflow's input / a gate's result are arbitrary JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import call_single, render_paginated
from aios.cli.coverage import covers
from aios.cli.files import PayloadError, load_payload
from aios.cli.runtime import get_state, run_or_die
from aios_sdk import stream_run
from aios_sdk._generated.api.runs import (
    cancel_run,
    create_run,
    get_run,
    list_run_events,
    list_runs,
    resume_gate,
)
from aios_sdk._generated.api.workflows import (
    create_workflow,
    get_workflow,
    list_workflows,
)
from aios_sdk._generated.models.gate_resume import GateResume
from aios_sdk._generated.models.wf_run_create import WfRunCreate
from aios_sdk._generated.models.workflow_create import WorkflowCreate

app = typer.Typer(name="workflows", help="Manage workflow definitions.", no_args_is_help=True)
runs_app = typer.Typer(name="runs", help="Launch and observe workflow runs.", no_args_is_help=True)


def _json_arg(value: str | None) -> Any:
    """Parse a JSON CLI argument (``--input`` / ``--result``); ``None`` stays ``None``.

    A malformed value raises :class:`PayloadError` so ``run_or_die`` prints a clean
    one-line message + exit 64, rather than letting ``json.loads`` bubble a traceback.
    """
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise PayloadError(f"invalid JSON: {exc}") from exc


# ─── aios workflows (definitions) ────────────────────────────────────────────


@app.command("list", help="List workflow definitions.")
@covers("list_workflows")
def list_workflows_(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Option("--name", help="Filter by exact name.")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    all_: Annotated[bool, typer.Option("--all", help="Fetch every page.")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_workflows.sync_detailed,
            columns=("id", "name", "version", "updated_at"),
            max_widths={"name": 32},
            all_=all_,
            limit=limit,
            name=name,
        )

    run_or_die(_run)


@app.command("get", help="Fetch a workflow definition by id.")
@covers("get_workflow")
def get_workflow_(ctx: typer.Context, workflow_id: str) -> None:
    def _run() -> None:
        call_single(ctx, get_workflow.sync_detailed, workflow_id=workflow_id)

    run_or_die(_run)


@app.command("create", help="Create a workflow from a JSON payload (WorkflowCreate shape).")
@covers("create_workflow")
def create_workflow_(
    ctx: typer.Context,
    file: Annotated[Path | None, typer.Option("--file", help="Read JSON body from a file.")] = None,
    stdin: Annotated[bool, typer.Option("--stdin", help="Read JSON body from stdin.")] = False,
    data: Annotated[str | None, typer.Option("--data", help="Inline JSON body.")] = None,
) -> None:
    def _run() -> int | None:
        payload = load_payload(file, stdin, data)
        call_single(ctx, create_workflow.sync_detailed, body=WorkflowCreate.from_dict(payload))
        return None

    run_or_die(_run)


# ─── aios runs (executions) ──────────────────────────────────────────────────


@runs_app.command("list", help="List workflow runs.")
@covers("list_runs")
def list_runs_(
    ctx: typer.Context,
    workflow_id: Annotated[str | None, typer.Option("--workflow-id")] = None,
    status: Annotated[str | None, typer.Option("--status", help="pending|running|…")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    all_: Annotated[bool, typer.Option("--all", help="Fetch every page.")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_runs.sync_detailed,
            columns=("id", "workflow_id", "status", "updated_at"),
            all_=all_,
            limit=limit,
            workflow_id=workflow_id,
            status=status,
        )

    run_or_die(_run)


@runs_app.command("get", help="Fetch a run by id.")
@covers("get_run")
def get_run_(ctx: typer.Context, run_id: str) -> None:
    def _run() -> None:
        call_single(ctx, get_run.sync_detailed, run_id=run_id)

    run_or_die(_run)


@runs_app.command("create", help="Launch a run of a workflow.")
@covers("create_run")
def create_run_(
    ctx: typer.Context,
    workflow_id: Annotated[str, typer.Option("--workflow-id")],
    environment_id: Annotated[str, typer.Option("--environment-id")],
    input_: Annotated[str | None, typer.Option("--input", help="The run's input as JSON.")] = None,
) -> None:
    def _run() -> int | None:
        body = WfRunCreate.from_dict(
            {
                "workflow_id": workflow_id,
                "environment_id": environment_id,
                "input": _json_arg(input_),
            }
        )
        call_single(ctx, create_run.sync_detailed, body=body)
        return None

    run_or_die(_run)


@runs_app.command("events", help="List a run's journal events (oldest first).")
@covers("list_run_events")
def run_events_(
    ctx: typer.Context,
    run_id: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=500)] = 200,
    all_: Annotated[bool, typer.Option("--all", help="Fetch every page.")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_run_events.sync_detailed,
            columns=("seq", "type", "call_key"),
            all_=all_,
            limit=limit,
            path_params={"run_id": run_id},
        )

    run_or_die(_run)


@runs_app.command("cancel", help="Cancel a run (it finalizes 'cancelled' on its next wake).")
@covers("cancel_run")
def cancel_run_(ctx: typer.Context, run_id: str) -> None:
    def _run() -> None:
        call_single(ctx, cancel_run.sync_detailed, run_id=run_id)

    run_or_die(_run)


@runs_app.command("resume", help="Resume a suspended gate by its nonce.")
@covers("resume_gate")
def resume_run_(
    ctx: typer.Context,
    run_id: str,
    gate_nonce: Annotated[str, typer.Option("--gate-nonce", help="The gate's capability nonce.")],
    result: Annotated[
        str | None, typer.Option("--result", help="The resume value as JSON.")
    ] = None,
) -> None:
    def _run() -> int | None:
        body = GateResume.from_dict({"gate_nonce": gate_nonce, "result": _json_arg(result)})
        call_single(ctx, resume_gate.sync_detailed, run_id=run_id, body=body)
        return None

    run_or_die(_run)


@runs_app.command("stream", help="Tail a run's journal as Server-Sent Events.")
@covers("stream_run_events_v1_runs__run_id__stream_get")
def stream_run_(
    ctx: typer.Context,
    run_id: str,
    after_seq: Annotated[int, typer.Option("--after-seq", min=0)] = 0,
    raw: Annotated[
        bool, typer.Option("--raw", help="Print each SSE message as JSON (no formatting).")
    ] = False,
) -> None:
    def _run() -> None:
        client = get_state(ctx).sdk_client()
        with client, stream_run(client, run_id, after_seq=after_seq) as messages:
            for msg in messages:
                if raw:
                    sys.stdout.write(json.dumps({"event": msg.event, "data": msg.data}) + "\n")
                elif msg.event == "event":
                    e = json.loads(msg.data)
                    sys.stdout.write(
                        f"{e['seq']:<4} {e['type']:<14} {json.dumps(e['payload'])[:100]}\n"
                    )
                sys.stdout.flush()
                if msg.event == "done":
                    break

    run_or_die(_run)
