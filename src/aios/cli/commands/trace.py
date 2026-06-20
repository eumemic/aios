"""``aios trace <id>`` — one-call linear trace of a run/session tree (#1149).

A thin client over ``GET /v1/runs/{id}/trace`` / ``GET /v1/sessions/{id}/trace``:
it dispatches by the id prefix (``wfr_…`` → run, ``sess_…`` → session), renders
``depth`` as indentation in table mode (the canonical DFS-pre-order makes a
child's subtree contiguous under its parent), and supports ``--chronological``
(re-sort the same entries by ``timestamp``, dropping indentation and showing
depth as a number), ``--verbose``, and the global ``--format {table,json}``.
"""

from __future__ import annotations

import sys
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import with_client
from aios.cli.coverage import covers
from aios.cli.output import dim, print_json
from aios.cli.runtime import run_or_die
from aios.ids import servicer_kind

_STATE_GLYPH = {
    "ok": "✓",
    "errored": "✗",
    "cancelled": "⊘",
    "suspended": "⏸",
    "running": "…",
}


def register(app: typer.Typer) -> None:
    @app.command(
        "trace",
        help="Linear trace of a run/session tree: nodes + interleaved journals (#1149).",
    )
    @covers("get_run_trace")
    @covers("get_session_trace")
    def trace(
        ctx: typer.Context,
        resource_id: Annotated[str, typer.Argument(help="A run (wfr_…) or session (sess_…) id.")],
        chronological: Annotated[
            bool,
            typer.Option(
                "--chronological",
                help="Re-sort entries by timestamp (transaction-granular, approximate) "
                "instead of the canonical causal DFS pre-order.",
            ),
        ] = False,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", help="Lift the abbreviation filter to the full journal."),
        ] = False,
    ) -> None:
        def _run() -> None:
            state, client = with_client(ctx)
            path = _trace_path(resource_id)
            with client:
                resp = client.request("GET", path, params={"verbose": verbose})
            if state.output_format == "json":
                print_json(resp)
                return
            _render_tree(resp, chronological=chronological)

        run_or_die(_run)


def _trace_path(resource_id: str) -> str:
    try:
        kind = servicer_kind(resource_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return f"/v1/runs/{resource_id}/trace" if kind == "run" else f"/v1/sessions/{resource_id}/trace"


def _render_tree(resp: dict[str, Any], *, chronological: bool) -> None:
    entries: list[dict[str, Any]] = list(resp.get("entries", []))
    if chronological:
        # Re-sort by timestamp; show depth as a number rather than indentation.
        entries.sort(key=lambda e: (e.get("timestamp") or "", e.get("depth", 0)))
    sys.stdout.write(f"{resp.get('root_kind')} {resp.get('root_id')}\n")
    for e in entries:
        sys.stdout.write(_format_entry(e, chronological=chronological) + "\n")
    truncated = resp.get("truncated")
    if truncated:
        sys.stdout.write(
            dim(f"… truncated at {truncated.get('at_nodes')} nodes (node-count ceiling)\n")
        )


def _format_entry(e: dict[str, Any], *, chronological: bool) -> str:
    depth = int(e.get("depth", 0))
    state = e.get("terminal_state")
    glyph = _STATE_GLYPH.get(state, " ") if state else " "
    error_kind = e.get("error_kind")
    suffix = f" [{error_kind}]" if error_kind else ""
    label = f"{e.get('kind')}: {e.get('summary', '')}".rstrip()
    if chronological:
        return f"d{depth} {glyph} {label}{suffix}"
    indent = "  " * depth
    return f"{indent}{glyph} {label}{suffix}"
