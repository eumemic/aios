"""``aios sessions ...`` — CRUD, send, interrupt, events, stream, tool-result/confirm."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import (
    get_state_and_client,
    raw_paginate,
    raw_paginate_events,
    raw_single,
    render_list,
    render_single,
)
from aios.cli.coverage import covers
from aios.cli.files import load_json_object, load_payload
from aios.cli.output import cyan, dim, print_error, print_json, print_success
from aios.cli.profile import compute_profile, profile_to_dict, render_profile
from aios.cli.runtime import get_state, run_or_die
from aios.cli.tail_format import iter_formatted_events
from aios_sdk import raw_request, stream_session

app = typer.Typer(name="sessions", help="Manage sessions.", no_args_is_help=True)

_SESSION_COLS = ("id", "agent_id", "status", "title", "updated_at")
_SESSION_MAXW = {"title": 32}


@app.command("list", help="List sessions.")
@covers("list_sessions")
def list_(
    ctx: typer.Context,
    agent_id: Annotated[str | None, typer.Option("--agent-id")] = None,
    status_filter: Annotated[
        str | None,
        typer.Option("--status", help="Filter by status: active, idle, archived."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = get_state_and_client(ctx)
        params: dict[str, Any] = {"agent_id": agent_id, "status": status_filter}
        with client:
            if all_:
                envelope = raw_paginate(client, "/v1/sessions", params=params)
            else:
                envelope = raw_request(
                    client,
                    "GET",
                    "/v1/sessions",
                    params={**params, "limit": limit},
                )
        render_list(state.output_format, envelope, columns=_SESSION_COLS, max_widths=_SESSION_MAXW)

    run_or_die(_run)


@app.command("get", help="Fetch a session by id.")
@covers("get_session")
def get(ctx: typer.Context, session_id: str) -> None:
    def _run() -> None:
        raw_single(ctx, "GET", f"/v1/sessions/{session_id}")

    run_or_die(_run)


@app.command("create", help="Create a session (SessionCreate shape).")
@covers("create_session")
def create(
    ctx: typer.Context,
    agent_id: Annotated[str | None, typer.Option("--agent", help="Agent id to bind.")] = None,
    environment_id: Annotated[str | None, typer.Option("--environment-id")] = None,
    title: Annotated[str | None, typer.Option("--title")] = None,
    message: Annotated[
        str | None, typer.Option("--message", help="Initial user message (appends + wakes).")
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        if any([file, stdin, data]):
            payload = load_payload(file, stdin, data)
        else:
            if agent_id is None or environment_id is None:
                print_error(
                    "either provide --agent and --environment-id, or supply a full payload "
                    "via --file/--stdin/--data."
                )
                return 64
            payload = {"agent_id": agent_id, "environment_id": environment_id}
            if title is not None:
                payload["title"] = title
            if message is not None:
                payload["initial_message"] = message
        raw_single(ctx, "POST", "/v1/sessions", json_body=payload)
        return None

    run_or_die(_run)


@app.command("update", help="Update a session (SessionUpdate shape).")
@covers("update_session")
def update(
    ctx: typer.Context,
    session_id: str,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        payload = load_payload(file, stdin, data)
        raw_single(ctx, "PUT", f"/v1/sessions/{session_id}", json_body=payload)
        return None

    run_or_die(_run)


@app.command("archive", help="Archive a session (soft delete via archived_at).")
@covers("archive_session")
def archive(ctx: typer.Context, session_id: str) -> None:
    def _run() -> None:
        raw_single(ctx, "POST", f"/v1/sessions/{session_id}/archive")

    run_or_die(_run)


@app.command(
    "clone",
    help="Clone a session — copy its event log into a new session id.",
)
@covers("clone_session")
def clone(
    ctx: typer.Context,
    session_id: str,
    count: Annotated[
        int,
        typer.Option("--count", min=1, max=100, help="Create N clones; print each new session."),
    ] = 1,
    workspace_path: Annotated[
        str | None,
        typer.Option(
            "--workspace-path",
            help="Override the clone's workspace volume path (must already exist). "
            "Mutually exclusive with --count > 1.",
        ),
    ] = None,
) -> None:
    def _run() -> int | None:
        if count > 1 and workspace_path is not None:
            print_error("--workspace-path cannot be combined with --count > 1")
            return 64
        body: dict[str, Any] = {}
        if workspace_path is not None:
            body["workspace_path"] = workspace_path
        results: list[dict[str, Any]] = []
        with get_state(ctx).sdk_client() as client:
            for _ in range(count):
                obj = raw_request(
                    client, "POST", f"/v1/sessions/{session_id}/clone", json_body=body
                )
                results.append(obj)
        if count == 1:
            render_single(results[0])
        else:
            envelope = {"data": results, "has_more": False, "next_cursor": None}
            state = get_state(ctx)
            render_list(
                state.output_format, envelope, columns=_SESSION_COLS, max_widths=_SESSION_MAXW
            )
        return None

    run_or_die(_run)


@app.command(
    "delete",
    help="Soft-archive a session (bare DELETE = soft-archive; reversible). "
    "Use `purge` for the irreversible hard-delete.",
)
@covers("delete_session")
def delete(ctx: typer.Context, session_id: str) -> None:
    def _run() -> None:
        raw_single(ctx, "DELETE", f"/v1/sessions/{session_id}")

    run_or_die(_run)


@app.command(
    "purge",
    help="Hard-delete a session and cascade its events/vaults/bindings "
    "(irreversible; prefer `archive`/`delete`).",
)
@covers("purge_session")
def purge(
    ctx: typer.Context,
    session_id: str,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Required to confirm hard-delete (no interactive prompt)."),
    ] = False,
) -> None:
    def _run() -> int | None:
        if not yes:
            print_error(
                "hard-delete is irreversible; pass --yes to confirm "
                "(or use `aios sessions delete` for a reversible soft-archive)"
            )
            return 2
        with get_state(ctx).sdk_client() as client:
            raw_request(client, "POST", f"/v1/sessions/{session_id}/purge")
        print_success("purged", session_id)
        return None

    run_or_die(_run)


@app.command("send", help="Append a user message and wake the session.")
@covers("send_message")
def send(
    ctx: typer.Context,
    session_id: str,
    message: Annotated[str, typer.Argument(help="The message content.")],
    metadata: Annotated[
        str | None, typer.Option("--metadata", help="Optional JSON metadata object.")
    ] = None,
) -> None:
    def _run() -> int | None:
        body: dict[str, Any] = {"content": message}
        if metadata is not None:
            body["metadata"] = load_json_object(metadata, "--metadata")
        raw_single(ctx, "POST", f"/v1/sessions/{session_id}/messages", json_body=body)
        return None

    run_or_die(_run)


@app.command("interrupt", help="Interrupt any in-flight turn for a session.")
@covers("interrupt_session")
def interrupt(
    ctx: typer.Context,
    session_id: str,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
) -> None:
    def _run() -> None:
        body: dict[str, Any] = {}
        if reason is not None:
            body["reason"] = reason
        raw_single(ctx, "POST", f"/v1/sessions/{session_id}/interrupt", json_body=body)

    run_or_die(_run)


@app.command("recycle-sandbox", help="Discard sandbox-local state and provision fresh.")
@covers("recycle_session_sandbox")
def recycle_sandbox(ctx: typer.Context, session_id: str) -> None:
    """Recycle while preserving bind-mounted durable session data."""

    def _run() -> None:
        raw_single(
            ctx,
            "POST",
            f"/v1/sessions/{session_id}/sandbox/recycle",
            json_body={"discard_unsalvaged": True},
        )

    run_or_die(_run)


@app.command("events", help="List a session's events (backfill).")
@covers("list_session_events")
def events(
    ctx: typer.Context,
    session_id: str,
    direction: Annotated[
        str,
        typer.Option("--dir", help="forward (oldest first) or backward (newest-first tail)."),
    ] = "forward",
    kind: Annotated[
        str | None,
        typer.Option("--kind", help="Filter by kind: message, lifecycle, span, interrupt."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=500)] = 200,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = get_state_and_client(ctx)
        with client:
            if all_:
                data = raw_paginate_events(client, session_id, kind=kind, direction=direction)
                envelope: dict[str, Any] = {
                    "data": data,
                    "has_more": False,
                    "next_cursor": None,
                }
            else:
                envelope = raw_request(
                    client,
                    "GET",
                    f"/v1/sessions/{session_id}/events",
                    params={"dir": direction, "kind": kind, "limit": limit},
                )
        render_list(
            state.output_format,
            envelope,
            columns=("seq", "kind", "created_at"),
        )

    run_or_die(_run)


@app.command(
    "profile",
    help="Per-phase latency breakdown of a session's span events.",
)
def profile(
    ctx: typer.Context,
    session_id: str,
    turns: Annotated[
        int | None,
        typer.Option(
            "--turns",
            min=1,
            help="Restrict to the last N turns (a turn starts at a user- or time-initiated wake).",
        ),
    ] = None,
) -> None:
    def _run() -> None:
        state, client = get_state_and_client(ctx)
        with client:
            events_data = raw_paginate_events(client, session_id, kind="span")

        result = compute_profile(events_data, turns=turns)
        if state.output_format == "json":
            print_json(profile_to_dict(result))
        else:
            sys.stdout.write(render_profile(result))

    run_or_die(_run)


@app.command("stream", help="Tail a session as Server-Sent Events.")
@covers("stream_events_v1_sessions__session_id__stream_get")
def stream(
    ctx: typer.Context,
    session_id: str,
    after_seq: Annotated[int, typer.Option("--after-seq", min=0)] = 0,
    raw: Annotated[
        bool,
        typer.Option("--raw", help="Print each SSE message as JSON (no event pretty-printing)."),
    ] = False,
    pretty: Annotated[
        bool,
        typer.Option(
            "--pretty",
            help="Use the structured one-line formatter (same as `aios tail`). "
            "Skips delta events and lifecycle-span noise.",
        ),
    ] = False,
) -> None:
    def _run() -> None:
        sys.stdout.write(dim(f"streaming {cyan(session_id)} after_seq={after_seq}\n"))
        if pretty:
            tail_session(ctx, session_id, after_seq=after_seq)
            return
        client = get_state(ctx).sdk_client()
        with client, stream_session(client, session_id, after_seq=after_seq) as messages:
            for msg in messages:
                if raw:
                    sys.stdout.write(json.dumps({"event": msg.event, "data": msg.data}) + "\n")
                    sys.stdout.flush()
                    continue
                _render_sse(msg.event, msg.data)
                if msg.event == "done":
                    break

    run_or_die(_run)


@app.command(
    "tail",
    help="Structured one-line real-time viewer — equivalent to `aios tail`.",
)
def tail(
    ctx: typer.Context,
    session_id: str,
    after_seq: Annotated[int, typer.Option("--after-seq", min=0)] = 0,
) -> None:
    def _run() -> None:
        tail_session(ctx, session_id, after_seq=after_seq)

    run_or_die(_run)


def tail_session(ctx: typer.Context, session_id: str, *, after_seq: int) -> None:
    """Shared implementation for ``aios sessions tail`` / top-level ``aios tail``."""
    client = get_state(ctx).sdk_client()
    with client, stream_session(client, session_id, after_seq=after_seq) as messages:
        for line in iter_formatted_events(messages):
            sys.stdout.write(line + "\n")
            sys.stdout.flush()


def _render_sse(event: str, data: str) -> None:
    """Pretty-print one SSE message to stdout for ``sessions stream``."""
    if event == "delta":
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return
        chunk = obj.get("delta")
        if isinstance(chunk, str):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        return
    if event == "done":
        sys.stdout.write(dim("\n[done]\n"))
        sys.stdout.flush()
        return
    if event == "event":
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            sys.stdout.write(data + "\n")
            return
        _render_event_obj(obj)


def _render_event_obj(obj: dict[str, Any]) -> None:
    kind = obj.get("kind")
    data = obj.get("data") or {}
    seq = obj.get("seq")
    if kind == "message":
        role = data.get("role")
        content = data.get("content") or ""
        tool_calls = data.get("tool_calls") or []
        if role == "user":
            sys.stdout.write(f"\n{dim(f'[{seq}]')} {cyan('user:')} {content}\n")
        elif role == "assistant":
            if content:
                sys.stdout.write(f"\n{dim(f'[{seq}]')} {cyan('assistant:')} {content}\n")
            for call in tool_calls:
                name = (call.get("function") or {}).get("name", "?")
                args = (call.get("function") or {}).get("arguments", "")
                sys.stdout.write(f"  → {name}({_compact(args, 200)})\n")
        elif role == "tool":
            name = data.get("tool_name") or data.get("name") or "tool"
            sys.stdout.write(f"  ← {name}: {_compact(content, 200)}\n")
        else:
            sys.stdout.write(f"{dim(f'[{seq}]')} message {json.dumps(data)}\n")
        return
    if kind == "lifecycle":
        ev = data.get("event", "")
        status = data.get("status", "")
        stop = data.get("stop_reason", "")
        bits = [f"kind=lifecycle event={ev}"]
        if status:
            bits.append(f"status={status}")
        if stop:
            bits.append(f"stop_reason={stop}")
        sys.stdout.write(f"{dim(f'[{seq}] ' + ' '.join(bits))}\n")
        return
    if kind == "span":
        ev = data.get("event", "")
        sys.stdout.write(f"{dim(f'[{seq}] span {ev}')}\n")
        return
    if kind == "interrupt":
        reason = data.get("reason") or ""
        sys.stdout.write(f"{dim(f'[{seq}] interrupt reason={reason}')}\n")
        return
    sys.stdout.write(f"{dim(f'[{seq}]')} {json.dumps(obj)}\n")


def _compact(value: Any, limit: int) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(s) > limit:
        return s[: limit - 1] + "…"
    return s


@app.command("tool-result", help="Submit a custom-tool result for a pending tool call.")
@covers("submit_tool_result")
def tool_result(
    ctx: typer.Context,
    session_id: str,
    call_id: Annotated[str, typer.Option("--call-id", help="The tool_call_id.")],
    content: Annotated[str, typer.Option("--content", help="Tool result body (string).")],
    error: Annotated[bool, typer.Option("--error", help="Mark the result as a failure.")] = False,
) -> None:
    def _run() -> None:
        body = {"tool_call_id": call_id, "content": content, "is_error": error}
        raw_single(ctx, "POST", f"/v1/sessions/{session_id}/tool-results", json_body=body)

    run_or_die(_run)


@app.command("tool-confirm", help="Confirm or deny an always_ask built-in tool call.")
@covers("submit_tool_confirmation")
def tool_confirm(
    ctx: typer.Context,
    session_id: str,
    call_id: Annotated[str, typer.Option("--call-id")],
    allow: Annotated[bool, typer.Option("--allow", help="Allow the tool call.")] = False,
    deny: Annotated[bool, typer.Option("--deny", help="Deny the tool call.")] = False,
    message: Annotated[str | None, typer.Option("--message", help="Optional deny message.")] = None,
) -> None:
    def _run() -> int | None:
        if allow == deny:
            print_error("exactly one of --allow or --deny must be provided")
            return 64
        body: dict[str, Any] = {
            "tool_call_id": call_id,
            "result": "allow" if allow else "deny",
        }
        if message is not None:
            body["deny_message"] = message
        raw_single(ctx, "POST", f"/v1/sessions/{session_id}/tool-confirmations", json_body=body)
        return None

    run_or_die(_run)


# ─── triggers sub-app ──────────────────────────────────────────────────────


triggers_app = typer.Typer(
    name="triggers",
    help="Manage a session's triggers (a source that fires + an action that runs).",
    no_args_is_help=True,
)
app.add_typer(triggers_app)

_TRIGGER_COLS = ("id", "name", "source", "action", "enabled", "next_fire", "last_fire_status")


def _source_summary(row: dict[str, Any]) -> str:
    """Synthesize the ``source`` cell for a trigger row.

    ``wake: <reason>`` for one-shot rows tagged ``metadata.kind == "wake"``
    (created by ``schedule_wake``); otherwise ``cron: <expr>`` or
    ``once @ <fire_at>`` from the source object.
    """
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("kind") == "wake":
        reason = metadata.get("reason") or ""
        return f"wake: {reason}" if reason else "wake"
    source = row.get("source") or {}
    if isinstance(source, dict):
        if source.get("kind") == "cron":
            return f"cron: {source.get('schedule')}"
        if source.get("kind") == "one_shot":
            return f"once @ {source.get('fire_at')}"
        if source.get("kind") == "run_completion":
            statuses = source.get("statuses") or []
            return f"on-completion: {source.get('workflow_id')} {statuses}"
    return ""


def _action_summary(row: dict[str, Any]) -> str:
    """Synthesize the ``action`` cell: kind + a short command/content preview."""
    action = row.get("action") or {}
    if not isinstance(action, dict):
        return ""
    kind = action.get("kind")
    if kind == "sandbox_command":
        return f"sandbox: {_preview(action.get('command') or '')}"
    if kind == "wake_owner":
        return f"wake_owner: {_preview(action.get('content') or '')}"
    if kind == "workflow":
        sel = action.get("version")
        pin = action.get("workflow_version")
        if sel is not None:
            suffix = f" =v{sel}"  # selector: re-run a historical version
        elif pin is not None:
            suffix = f" @v{pin}"  # drift assertion
        else:
            suffix = ""
        return f"workflow: {action.get('workflow_id')}{suffix}"
    return str(kind or "")


def _preview(text: str, limit: int = 40) -> str:
    # Collapse newlines (a literal newline breaks the table cell) then defer
    # to the shared truncator so we don't grow a fourth ellipsis helper.
    return _compact(text.replace("\n", " "), limit)


def _build_source(cron: str | None, at: str | None) -> dict[str, Any] | None:
    """One of ``--cron`` / ``--at`` → a source object; None if not exactly one."""
    if (cron is None) == (at is None):
        return None
    if cron is not None:
        return {"kind": "cron", "schedule": cron}
    return {"kind": "one_shot", "fire_at": at}


def _build_action(command: str | None, wake_content: str | None) -> dict[str, Any] | None:
    """One of ``--command`` / ``--wake-content`` → an action object."""
    if (command is None) == (wake_content is None):
        return None
    if command is not None:
        return {"kind": "sandbox_command", "command": command}
    return {"kind": "wake_owner", "content": wake_content}


@triggers_app.command("list", help="List triggers for a session.")
@covers("list_triggers")
def triggers_list(ctx: typer.Context, session_id: str) -> None:
    def _run() -> None:
        state, client = get_state_and_client(ctx)
        with client:
            envelope = raw_request(client, "GET", f"/v1/sessions/{session_id}/triggers")
        # Annotate each row with the synthesized source/action cells before
        # the renderer reads its columns. JSON output keeps the raw shape.
        if state.output_format != "json":
            data = envelope.get("data") or []
            for row in data:
                if isinstance(row, dict):
                    row["source"] = _source_summary(row)
                    row["action"] = _action_summary(row)
        render_list(state.output_format, envelope, columns=_TRIGGER_COLS)

    run_or_die(_run)


@triggers_app.command("add", help="Add a trigger to a session.")
@covers("create_trigger")
def triggers_add(
    ctx: typer.Context,
    session_id: str,
    name: Annotated[str | None, typer.Option("--name")] = None,
    cron: Annotated[str | None, typer.Option("--cron", help="Cron expression (source).")] = None,
    at: Annotated[
        str | None, typer.Option("--at", help="One-shot ISO 8601 fire time (source).")
    ] = None,
    command: Annotated[str | None, typer.Option("--command", help="Bash command (action).")] = None,
    wake_content: Annotated[
        str | None, typer.Option("--wake-content", help="wake_owner message (action).")
    ] = None,
    enabled: Annotated[bool, typer.Option("--enabled/--disabled")] = True,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        if any([file, stdin, data]):
            payload = load_payload(file, stdin, data)
        else:
            if not name:
                print_error(
                    "provide --name plus a source (--cron or --at) and an action "
                    "(--command or --wake-content), or a full payload via "
                    "--file/--stdin/--data."
                )
                return 64
            source = _build_source(cron, at)
            if source is None:
                print_error("provide exactly one source: --cron <expr> or --at <iso>.")
                return 64
            action = _build_action(command, wake_content)
            if action is None:
                print_error(
                    "provide exactly one action: --command <bash> or --wake-content <text>."
                )
                return 64
            payload = {"name": name, "source": source, "action": action, "enabled": enabled}
        raw_single(ctx, "POST", f"/v1/sessions/{session_id}/triggers", json_body=payload)
        return None

    run_or_die(_run)


@triggers_app.command("remove", help="Remove a trigger by name.")
@covers("delete_trigger")
def triggers_remove(ctx: typer.Context, session_id: str, name: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            raw_request(client, "DELETE", f"/v1/sessions/{session_id}/triggers/{name}")
        print_success("removed", name)

    run_or_die(_run)


@triggers_app.command("update", help="Update a trigger by name.")
@covers("update_trigger")
def triggers_update(
    ctx: typer.Context,
    session_id: str,
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
        else:
            payload = {}
            if enabled is not None:
                payload["enabled"] = enabled
            if not payload:
                print_error(
                    "provide --enabled/--disabled, or a full payload via "
                    "--file/--stdin/--data (source/action replace the stored object "
                    "wholesale — fetch current values via `triggers list`)."
                )
                return 64
        raw_single(ctx, "PUT", f"/v1/sessions/{session_id}/triggers/{name}", json_body=payload)
        return None

    run_or_die(_run)


_TRIGGER_RUN_COLS = (
    "id",
    "trigger_context",
    "status",
    "result_id",
    "error_summary",
    "created_at",
    "started_at",
    "finished_at",
)


@triggers_app.command("runs", help="List a trigger's fires (the per-fire audit), newest first.")
@covers("list_trigger_runs")
def triggers_runs(
    ctx: typer.Context,
    session_id: str,
    name: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
) -> None:
    def _run() -> None:
        state, client = get_state_and_client(ctx)
        with client:
            envelope = raw_request(
                client,
                "GET",
                f"/v1/sessions/{session_id}/triggers/{name}/runs",
                params={"limit": limit},
            )
        render_list(state.output_format, envelope, columns=_TRIGGER_RUN_COLS)

    run_or_die(_run)


# ─── resources sub-app ─────────────────────────────────────────────────────


resources_app = typer.Typer(
    name="resources",
    help="Manage a session's resources (memory stores + github repositories).",
    no_args_is_help=True,
)
app.add_typer(resources_app)

_RESOURCE_COLS = ("type", "memory_store_id", "id", "name", "url", "mount_path")


@resources_app.command("list", help="List resources attached to a session.")
@covers("list_session_resources")
def resources_list(ctx: typer.Context, session_id: str) -> None:
    def _run() -> None:
        state, client = get_state_and_client(ctx)
        with client:
            envelope = raw_request(client, "GET", f"/v1/sessions/{session_id}/resources")
        render_list(state.output_format, envelope, columns=_RESOURCE_COLS)

    run_or_die(_run)


@resources_app.command("add", help="Attach a single resource to a session (additive, #270).")
@covers("add_session_resource")
def resources_add(
    ctx: typer.Context,
    session_id: str,
    memory_store_id: Annotated[
        str | None,
        typer.Option("--memory-store-id", help="Attach this memory store (memory_store)."),
    ] = None,
    access: Annotated[
        str, typer.Option("--access", help="Memory store access: read_write or read_only.")
    ] = "read_write",
    instructions: Annotated[
        str, typer.Option("--instructions", help="Memory store usage instructions.")
    ] = "",
    url: Annotated[
        str | None, typer.Option("--url", help="Github repository clone URL (github_repository).")
    ] = None,
    mount_path: Annotated[
        str | None, typer.Option("--mount-path", help="Where to clone the github repository.")
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", help="Github authorization token (write-only)."),
    ] = None,
    git_user_name: Annotated[str | None, typer.Option("--git-user-name")] = None,
    git_user_email: Annotated[str | None, typer.Option("--git-user-email")] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        if any([file, stdin, data]):
            payload = load_payload(file, stdin, data)
        elif memory_store_id is not None:
            payload = {
                "type": "memory_store",
                "memory_store_id": memory_store_id,
                "access": access,
                "instructions": instructions,
            }
        elif url is not None or mount_path is not None or token is not None:
            if not (url and mount_path and token):
                print_error("github_repository requires --url, --mount-path, and --token together.")
                return 64
            payload = {
                "type": "github_repository",
                "url": url,
                "mount_path": mount_path,
                "authorization_token": token,
            }
            if git_user_name is not None:
                payload["git_user_name"] = git_user_name
            if git_user_email is not None:
                payload["git_user_email"] = git_user_email
        else:
            print_error(
                "provide --memory-store-id (memory store), or --url/--mount-path/--token "
                "(github repository), or a full payload via --file/--stdin/--data."
            )
            return 64
        raw_single(ctx, "POST", f"/v1/sessions/{session_id}/resources", json_body=payload)
        return None

    run_or_die(_run)


@resources_app.command("remove", help="Detach a single resource by id (additive, #270).")
@covers("remove_session_resource")
def resources_remove(ctx: typer.Context, session_id: str, resource_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            raw_request(client, "DELETE", f"/v1/sessions/{session_id}/resources/{resource_id}")
        print_success("removed", resource_id)

    run_or_die(_run)


@resources_app.command("rotate", help="Rotate a github_repository's auth token by id.")
@covers("update_session_resource")
def resources_rotate(
    ctx: typer.Context,
    session_id: str,
    resource_id: str,
    token: Annotated[
        str | None, typer.Option("--token", help="New github authorization token.")
    ] = None,
    git_user_name: Annotated[str | None, typer.Option("--git-user-name")] = None,
    git_user_email: Annotated[str | None, typer.Option("--git-user-email")] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        if any([file, stdin, data]):
            payload = load_payload(file, stdin, data)
        elif token is not None:
            payload = {"authorization_token": token}
            if git_user_name is not None:
                payload["git_user_name"] = git_user_name
            if git_user_email is not None:
                payload["git_user_email"] = git_user_email
        else:
            print_error(
                "provide --token (and optionally --git-user-name/--git-user-email), "
                "or a full payload via --file/--stdin/--data."
            )
            return 64
        raw_single(
            ctx,
            "PUT",
            f"/v1/sessions/{session_id}/resources/{resource_id}",
            json_body=payload,
        )
        return None

    run_or_die(_run)
