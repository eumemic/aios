"""``aios chat`` — interactive REPL that streams a session's replies.

Flow per turn:

1. Read a user message from stdin. Blank input re-prompts; ``/exit`` quits;
   ``/interrupt`` posts an interrupt; ``/help`` prints command help.
2. ``POST /v1/sessions/{id}/messages`` to append the message.
3. Open the SSE stream from the last seq we've seen and render events until
   a turn-end lifecycle (``status == "idle"``) or a ``done`` SSE arrives.
4. Return to the prompt.

Only standard library + httpx + ANSI escapes — no rich/prompt_toolkit.
"""

from __future__ import annotations

import json
import signal
import sys
from contextlib import suppress
from types import FrameType
from typing import Annotated, Any

import typer

from aios.cli.client import AiosApiError, AiosClient
from aios.cli.output import bold, cyan, dim, green, print_error, yellow
from aios.cli.runtime import get_state, run_or_die


def register(app: typer.Typer) -> None:
    @app.command(
        "chat",
        help=(
            "Interactive REPL against a session. Pass --agent to create a new session, "
            "or --session to join an existing one."
        ),
    )
    def chat(
        ctx: typer.Context,
        agent: Annotated[
            str | None, typer.Option("--agent", help="Agent id (creates a new session).")
        ] = None,
        environment_id: Annotated[
            str | None,
            typer.Option(
                "--environment-id",
                help="Environment id (required when creating a new session).",
            ),
        ] = None,
        session: Annotated[
            str | None, typer.Option("--session", help="Existing session id to join.")
        ] = None,
        title: Annotated[str | None, typer.Option("--title")] = None,
        initial: Annotated[
            str | None,
            typer.Option(
                "--initial",
                "-m",
                help="Send this message immediately as the first turn (non-interactive).",
            ),
        ] = None,
    ) -> None:
        def _run() -> int | None:
            if agent is None and session is None:
                print_error(
                    "chat requires either --agent AGENT_ID (to create a session) or "
                    "--session SESSION_ID (to join one)"
                )
                return 64
            if agent is not None and session is not None:
                print_error("pass only one of --agent or --session")
                return 64
            state = get_state(ctx)
            client = state.client()
            try:
                session_id = _resolve_session(
                    client,
                    agent=agent,
                    environment_id=environment_id,
                    session=session,
                    title=title,
                )
                _run_repl(client, session_id, initial=initial, verbose=state.verbose)
            finally:
                client.close()
            return None

        run_or_die(_run)


def _resolve_session(
    client: AiosClient,
    *,
    agent: str | None,
    environment_id: str | None,
    session: str | None,
    title: str | None,
) -> str:
    if session is not None:
        obj = client.request("GET", f"/v1/sessions/{session}")
        sys.stdout.write(
            f"joined session {cyan(obj['id'])} (agent={obj['agent_id']}, status={obj['status']})\n"
        )
        return str(obj["id"])

    assert agent is not None
    if environment_id is None:
        raise AiosApiError(
            status_code=0,
            error_type="usage_error",
            message="--environment-id is required when creating a new session with --agent",
        )
    body: dict[str, Any] = {"agent_id": agent, "environment_id": environment_id}
    if title is not None:
        body["title"] = title
    obj = client.request("POST", "/v1/sessions", json_body=body)
    sys.stdout.write(
        f"created session {cyan(obj['id'])} (agent={obj['agent_id']}, "
        f"version={obj.get('agent_version')})\n"
    )
    return str(obj["id"])


def _run_repl(
    client: AiosClient,
    session_id: str,
    *,
    initial: str | None,
    verbose: bool,
) -> None:
    sys.stdout.write(
        dim(
            "type a message and press Enter. /exit quits, /interrupt cancels an in-flight turn, "
            "/help shows commands.\n"
        )
    )
    last_seq = 0

    # One-shot initial message: send, stream, then exit without prompting.
    if initial is not None:
        _post_message(client, session_id, initial)
        last_seq = _stream_until_idle(client, session_id, last_seq, verbose=verbose)
        return

    while True:
        try:
            line = input(bold("> "))
        except EOFError:
            sys.stdout.write("\n")
            return
        stripped = line.strip()
        if stripped == "":
            continue
        if stripped in {"/exit", "/quit"}:
            return
        if stripped == "/help":
            sys.stdout.write(
                dim(
                    "  /exit      quit the REPL\n"
                    "  /interrupt cancel the in-flight turn (posts to /interrupt)\n"
                    "  /help      show this help\n"
                )
            )
            continue
        if stripped == "/interrupt":
            with suppress(AiosApiError):
                client.request("POST", f"/v1/sessions/{session_id}/interrupt", json_body={})
            sys.stdout.write(yellow("  [interrupted]\n"))
            continue

        _post_message(client, session_id, stripped)
        last_seq = _stream_until_idle(client, session_id, last_seq, verbose=verbose)


def _post_message(client: AiosClient, session_id: str, content: str) -> None:
    client.request(
        "POST",
        f"/v1/sessions/{session_id}/messages",
        json_body={"content": content},
    )


def _stream_until_idle(
    client: AiosClient,
    session_id: str,
    last_seq: int,
    *,
    verbose: bool,
) -> int:
    """Consume SSE events until the current turn ends. Returns new last_seq."""

    def _handle_sigint(_sig: int, _frame: FrameType | None) -> None:
        # Typer re-raises KeyboardInterrupt via the default handler; we only
        # want to post an interrupt and let the stream close naturally.
        with suppress(AiosApiError):
            client.request("POST", f"/v1/sessions/{session_id}/interrupt", json_body={})
        sys.stderr.write(yellow("\n  [ctrl-c: interrupt requested]\n", stream=sys.stderr))

    previous_handler = signal.signal(signal.SIGINT, _handle_sigint)
    try:
        with client.stream_session(session_id, after_seq=last_seq) as messages:
            for msg in messages:
                if msg.event == "delta":
                    _write_delta(msg.data)
                    continue
                if msg.event == "done":
                    sys.stdout.write(dim("\n[session terminated]\n"))
                    return last_seq
                if msg.event != "event":
                    continue
                try:
                    obj = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                new_seq = obj.get("seq")
                if isinstance(new_seq, int) and new_seq > last_seq:
                    last_seq = new_seq
                _render_chat_event(obj, verbose=verbose)
                if _is_turn_end(obj):
                    sys.stdout.write(dim(f"  [turn ended at seq={last_seq}]\n"))
                    return last_seq
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    return last_seq


def _write_delta(data: str) -> None:
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return
    chunk = obj.get("delta")
    if isinstance(chunk, str):
        sys.stdout.write(chunk)
        sys.stdout.flush()


def _render_chat_event(obj: dict[str, Any], *, verbose: bool) -> None:
    kind = obj.get("kind")
    data = obj.get("data") or {}
    if kind == "message":
        role = data.get("role")
        content = data.get("content") or ""
        tool_calls = data.get("tool_calls") or []
        if role == "user":
            # We already echoed this from the prompt; don't re-print.
            return
        if role == "assistant":
            if content and not tool_calls:
                # Content may have already been streamed via deltas. The final
                # event also carries it — detect by checking if we've already
                # written non-empty text in this turn. Simpler: write a newline
                # to terminate the delta stream, and suppress the full-content
                # re-render. If no deltas came (e.g. non-streaming model),
                # write the content now.
                sys.stdout.write("\n")
                return
            for call in tool_calls:
                name = (call.get("function") or {}).get("name", "?")
                args = (call.get("function") or {}).get("arguments", "")
                sys.stdout.write(f"\n  {cyan('→')} {bold(name)}({_compact(args, 160, verbose)})\n")
            return
        if role == "tool":
            name = data.get("tool_name") or data.get("name") or "tool"
            is_err = bool(data.get("is_error"))
            arrow = yellow("←") if is_err else green("←")
            sys.stdout.write(f"  {arrow} {bold(name)}: {_compact(content, 240, verbose)}\n")
            return
        sys.stdout.write(dim(f"  [message role={role}]\n"))
        return
    if kind == "lifecycle":
        event_name = data.get("event", "")
        status = data.get("status", "")
        if event_name == "turn_started":
            return
        if event_name == "turn_ended":
            return
        if event_name == "interrupted":
            sys.stdout.write(yellow(f"  [{event_name}]\n"))
            return
        if verbose:
            sys.stdout.write(dim(f"  [lifecycle {event_name} status={status}]\n"))
        return
    if kind == "span":
        if verbose:
            sys.stdout.write(dim(f"  [span {data.get('event', '')}]\n"))
        return
    if kind == "interrupt":
        sys.stdout.write(yellow(f"  [interrupt {data.get('reason', '')}]\n"))
        return


def _is_turn_end(obj: dict[str, Any]) -> bool:
    """True if this event marks the end of the current turn."""
    if obj.get("kind") != "lifecycle":
        return False
    data = obj.get("data") or {}
    event_name = data.get("event")
    status = data.get("status")
    # turn_ended fires for both "end_turn" and "requires_action" cases.
    if event_name == "turn_ended":
        return True
    # Interrupt also returns us to the prompt.
    if event_name == "interrupted":
        return True
    # Terminated sessions end any pending turn.
    return status == "terminated"


def _compact(value: Any, limit: int, verbose: bool) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    if verbose or len(s) <= limit:
        return s
    return s[: limit - 1] + "…"
