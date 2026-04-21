"""``aios sessions ...`` — CRUD, send, interrupt, events, stream, tool-result/confirm."""

from __future__ import annotations

import json
import sys
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
from aios.cli.files import PayloadError, load_json_object, load_payload
from aios.cli.output import cyan, dim, print_error, print_success
from aios.cli.runtime import get_state, run_or_die
from aios.cli.tail_format import iter_formatted_events

app = typer.Typer(name="sessions", help="Manage sessions.", no_args_is_help=True)

_SESSION_COLS = ("id", "agent_id", "status", "title", "updated_at")
_SESSION_MAXW = {"title": 32}


@app.command("list", help="List sessions.")
def list_(
    ctx: typer.Context,
    agent_id: Annotated[str | None, typer.Option("--agent-id")] = None,
    status_filter: Annotated[
        str | None,
        typer.Option("--status", help="Filter by status: running, idle, terminated."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        params: dict[str, Any] = {"agent_id": agent_id, "status": status_filter}
        with client:
            if all_:
                envelope = fetch_all(client, "/v1/sessions", params=params)
            else:
                envelope = client.request(
                    "GET",
                    "/v1/sessions",
                    params={**params, "limit": limit, "after": after},
                )
        render_list(state.output_format, envelope, columns=_SESSION_COLS, max_widths=_SESSION_MAXW)

    run_or_die(_run)


@app.command("get", help="Fetch a session by id.")
def get(ctx: typer.Context, session_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/sessions/{session_id}")
        render_single(obj)

    run_or_die(_run)


@app.command("create", help="Create a session (SessionCreate shape).")
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
            try:
                payload = load_payload(file, stdin, data)
            except PayloadError as exc:
                print_error(str(exc))
                return 64
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
        client = just_client(ctx)
        with client:
            obj = client.request("POST", "/v1/sessions", json_body=payload)
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("update", help="Update a session (SessionUpdate shape).")
def update(
    ctx: typer.Context,
    session_id: str,
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
            obj = client.request("PUT", f"/v1/sessions/{session_id}", json_body=payload)
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("archive", help="Archive a session (status=terminated, soft).")
def archive(ctx: typer.Context, session_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("POST", f"/v1/sessions/{session_id}/archive")
        render_single(obj)

    run_or_die(_run)


@app.command(
    "delete",
    help="Hard-delete a session (irreversible; prefer `archive` instead).",
)
def delete(
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
                "(or use `aios sessions archive` for a reversible soft-delete)"
            )
            return 2
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/sessions/{session_id}")
        print_success("deleted", session_id)
        return None

    run_or_die(_run)


@app.command("send", help="Append a user message and wake the session.")
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
            try:
                body["metadata"] = load_json_object(metadata, "--metadata")
            except PayloadError as exc:
                print_error(str(exc))
                return 64
        client = just_client(ctx)
        with client:
            event = client.request("POST", f"/v1/sessions/{session_id}/messages", json_body=body)
        render_single(event)
        return None

    run_or_die(_run)


@app.command("interrupt", help="Interrupt any in-flight turn for a session.")
def interrupt(
    ctx: typer.Context,
    session_id: str,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
) -> None:
    def _run() -> None:
        client = just_client(ctx)
        body: dict[str, Any] = {}
        if reason is not None:
            body["reason"] = reason
        with client:
            obj = client.request("POST", f"/v1/sessions/{session_id}/interrupt", json_body=body)
        render_single(obj)

    run_or_die(_run)


@app.command("events", help="List a session's events (backfill).")
def events(
    ctx: typer.Context,
    session_id: str,
    after_seq: Annotated[int, typer.Option("--after-seq", min=0)] = 0,
    kind: Annotated[
        str | None,
        typer.Option("--kind", help="Filter by kind: message, lifecycle, span, interrupt."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 200,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        params: dict[str, Any] = {"after_seq": after_seq, "kind": kind}
        with client:
            if all_:
                # The events endpoint paginates by seq, not cursor id. Walk
                # pages manually, bumping after_seq to the last seen seq.
                data: list[dict[str, Any]] = []
                cursor_seq = after_seq
                while True:
                    page = client.request(
                        "GET",
                        f"/v1/sessions/{session_id}/events",
                        params={"after_seq": cursor_seq, "kind": kind, "limit": 200},
                    )
                    assert isinstance(page, dict)
                    page_data = page.get("data", [])
                    if not page_data:
                        break
                    data.extend(page_data)
                    last_seq = page_data[-1].get("seq")
                    if not page.get("has_more") or last_seq is None:
                        break
                    cursor_seq = int(last_seq)
                envelope = {"data": data, "has_more": False, "next_after": None}
            else:
                envelope = client.request(
                    "GET",
                    f"/v1/sessions/{session_id}/events",
                    params={**params, "limit": limit},
                )
        render_list(
            state.output_format,
            envelope,
            columns=("seq", "kind", "created_at"),
        )

    run_or_die(_run)


@app.command("stream", help="Tail a session as Server-Sent Events.")
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
        client = get_state(ctx).client()
        with client, client.stream_session(session_id, after_seq=after_seq) as messages:
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
    client = get_state(ctx).client()
    with client, client.stream_session(session_id, after_seq=after_seq) as messages:
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
def tool_result(
    ctx: typer.Context,
    session_id: str,
    call_id: Annotated[str, typer.Option("--call-id", help="The tool_call_id.")],
    content: Annotated[str, typer.Option("--content", help="Tool result body (string).")],
    error: Annotated[bool, typer.Option("--error", help="Mark the result as a failure.")] = False,
) -> None:
    def _run() -> None:
        client = just_client(ctx)
        body = {"tool_call_id": call_id, "content": content, "is_error": error}
        with client:
            event = client.request(
                "POST", f"/v1/sessions/{session_id}/tool-results", json_body=body
            )
        render_single(event)

    run_or_die(_run)


@app.command("tool-confirm", help="Confirm or deny an always_ask built-in tool call.")
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
        client = just_client(ctx)
        body: dict[str, Any] = {
            "tool_call_id": call_id,
            "result": "allow" if allow else "deny",
        }
        if message is not None:
            body["deny_message"] = message
        with client:
            event = client.request(
                "POST", f"/v1/sessions/{session_id}/tool-confirmations", json_body=body
            )
        render_single(event)
        return None

    run_or_die(_run)
