"""``aios connections ...`` — connector-instance CRUD plus mode-binding helpers.

Connections are created in detached mode; switch to single_session via
``attach`` or per_chat via ``configure-per-chat``.  ``archive`` works
only on detached connections — operators must ``detach`` /
``unconfigure`` first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import (
    fetch_all_sdk,
    render_list,
    render_sdk_list,
    render_single,
    unwrap,
)
from aios.cli.files import PayloadError, load_json_object, resolve_payload
from aios.cli.output import print_error, print_success
from aios.cli.runtime import get_state, run_or_die
from aios.sdk._generated.api.connections import (
    archive_connection,
    attach_connection,
    bind_chat,
    configure_connection_per_chat,
    create_connection,
    detach_connection,
    get_connection,
    list_bound_chats,
    list_connections,
    list_recent_chats,
    unbind_chat,
    unconfigure_connection,
)
from aios.sdk._generated.models.bind_chat_request import BindChatRequest
from aios.sdk._generated.models.connection_attach import ConnectionAttach
from aios.sdk._generated.models.connection_configure_per_chat import ConnectionConfigurePerChat
from aios.sdk._generated.models.connection_create import ConnectionCreate
from aios.sdk._generated.models.list_connections_mode_type_0 import ListConnectionsModeType0
from aios.sdk._generated.types import UNSET, Unset

app = typer.Typer(name="connections", help="Manage connector connections.", no_args_is_help=True)

_COLS = ("id", "connector", "account", "session_id", "session_template_id", "updated_at")
_MAXW = {"connector": 20, "account": 40, "session_id": 24, "session_template_id": 24}


def _coerce_mode(mode: str | None) -> ListConnectionsModeType0 | Unset:
    """Accept the typer-side string and project to the typed SDK enum.

    The SDK generates a per-operation enum (``ListConnectionsModeType0``)
    rather than reusing :class:`aios.models.connections.ConnectionMode`,
    so the surface here looks awkward — that's a generator artifact, not
    a contract decision.
    """
    if mode is None:
        return UNSET
    return ListConnectionsModeType0(mode)


@app.command("list")
def list_(
    ctx: typer.Context,
    connector: Annotated[str | None, typer.Option("--connector")] = None,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Filter by routing mode: detached, single_session, per_chat."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state = get_state(ctx)
        sdk_mode = _coerce_mode(mode)
        with state.sdk_client() as client:
            if all_:
                items = fetch_all_sdk(
                    list_connections.sync_detailed,
                    client=client,
                    connector=connector,
                    session_id=session_id,
                    mode=sdk_mode,
                )
                render_sdk_list(state.output_format, items, columns=_COLS, max_widths=_MAXW)
                return
            page = unwrap(
                list_connections.sync_detailed(
                    client=client,
                    connector=connector,
                    session_id=session_id,
                    mode=sdk_mode,
                    limit=limit,
                    after=after,
                )
            )
            render_list(state.output_format, page.to_dict(), columns=_COLS, max_widths=_MAXW)

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(get_connection.sync_detailed(client=client, connection_id=connection_id))
        render_single(obj.to_dict())

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
        body = ConnectionCreate.from_dict(payload)
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(create_connection.sync_detailed(client=client, body=body))
        render_single(obj.to_dict())
        return None

    run_or_die(_run)


@app.command("attach", help="Bind a detached connection to a session (single_session mode).")
def attach(
    ctx: typer.Context,
    connection_id: str,
    session_id: Annotated[str, typer.Option("--session-id")],
) -> None:
    def _run() -> None:
        body = ConnectionAttach(session_id=session_id)
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(
                attach_connection.sync_detailed(
                    client=client, connection_id=connection_id, body=body
                )
            )
        render_single(obj.to_dict())

    run_or_die(_run)


@app.command("detach", help="Drop the single_session binding, leaving the connection detached.")
def detach(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(
                detach_connection.sync_detailed(client=client, connection_id=connection_id)
            )
        render_single(obj.to_dict())

    run_or_die(_run)


@app.command("configure-per-chat", help="Switch the connection into per_chat mode.")
def configure_per_chat(
    ctx: typer.Context,
    connection_id: str,
    template: Annotated[str, typer.Option("--template", help="Session template id.")],
) -> None:
    def _run() -> None:
        body = ConnectionConfigurePerChat(session_template_id=template)
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(
                configure_connection_per_chat.sync_detailed(
                    client=client, connection_id=connection_id, body=body
                )
            )
        render_single(obj.to_dict())

    run_or_die(_run)


@app.command(
    "unconfigure", help="Drop the per_chat configuration, leaving the connection detached."
)
def unconfigure(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(
                unconfigure_connection.sync_detailed(client=client, connection_id=connection_id)
            )
        render_single(obj.to_dict())

    run_or_die(_run)


@app.command(
    "bind-chat",
    help="Pre-bind a chat_id on a connection's account to an existing session.",
)
def bind_chat_cmd(
    ctx: typer.Context,
    connection_id: str,
    chat_id: Annotated[str, typer.Option("--chat-id", help="Platform-native chat id.")],
    session_id: Annotated[str, typer.Option("--session-id", help="Target session id.")],
) -> None:
    def _run() -> None:
        body = BindChatRequest(chat_id=chat_id, session_id=session_id)
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(
                bind_chat.sync_detailed(client=client, connection_id=connection_id, body=body)
            )
        render_single(obj.to_dict())

    run_or_die(_run)


@app.command(
    "unbind-chat",
    help="Drop a chat → session binding, returning the chat to the connection's mode default.",
)
def unbind_chat_cmd(
    ctx: typer.Context,
    connection_id: str,
    chat_id: Annotated[str, typer.Option("--chat-id")],
) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            unwrap(
                unbind_chat.sync_detailed(
                    client=client, connection_id=connection_id, chat_id=chat_id
                )
            )
        print_success("unbound", f"{connection_id}:{chat_id}")

    run_or_die(_run)


@app.command(
    "bound-chats",
    help="List all chat → session bindings (operator-curated + supervisor-spawned).",
)
def bound_chats(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        state = get_state(ctx)
        with state.sdk_client() as client:
            page = unwrap(
                list_bound_chats.sync_detailed(client=client, connection_id=connection_id)
            )
        render_list(
            state.output_format,
            page.to_dict(),
            columns=("chat_id", "session_id", "created_at"),
            max_widths={"chat_id": 40, "session_id": 24},
        )

    run_or_die(_run)


@app.command(
    "recent-chats",
    help="List distinct chat_ids on this connection's account that have produced inbound.",
)
def recent_chats(
    ctx: typer.Context,
    connection_id: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
) -> None:
    def _run() -> None:
        state = get_state(ctx)
        with state.sdk_client() as client:
            page = unwrap(
                list_recent_chats.sync_detailed(
                    client=client, connection_id=connection_id, limit=limit
                )
            )
        render_list(
            state.output_format,
            page.to_dict(),
            columns=("chat_id", "last_seen_at"),
            max_widths={"chat_id": 40},
        )

    run_or_die(_run)


@app.command("archive", help="Archive a detached connection (soft-delete, retained for audit).")
def archive(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            unwrap(archive_connection.sync_detailed(client=client, connection_id=connection_id))
        print_success("archived", connection_id)

    run_or_die(_run)
