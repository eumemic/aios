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
    call_single,
    render_list,
    render_paginated,
    unwrap,
)
from aios.cli.files import PayloadError, load_json_object, resolve_payload
from aios.cli.output import print_error, print_success
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.connections import (
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
    set_connection_secrets,
    unbind_chat,
    unconfigure_connection,
)
from aios_sdk._generated.models.bind_chat_request import BindChatRequest
from aios_sdk._generated.models.connection_attach import ConnectionAttach
from aios_sdk._generated.models.connection_configure_per_chat import ConnectionConfigurePerChat
from aios_sdk._generated.models.connection_create import ConnectionCreate
from aios_sdk._generated.models.connection_set_secrets import ConnectionSetSecrets
from aios_sdk._generated.models.connection_set_secrets_secrets import ConnectionSetSecretsSecrets
from aios_sdk._generated.models.list_connections_mode_type_0 import ListConnectionsModeType0

app = typer.Typer(name="connections", help="Manage connector connections.", no_args_is_help=True)

_COLS = (
    "id",
    "connector",
    "external_account_id",
    "session_id",
    "session_template_id",
    "updated_at",
)
_MAXW = {
    "connector": 20,
    "external_account_id": 40,
    "session_id": 24,
    "session_template_id": 24,
}


@app.command("list")
def list_(
    ctx: typer.Context,
    connector: Annotated[str | None, typer.Option("--connector")] = None,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    mode: Annotated[
        ListConnectionsModeType0 | None,
        typer.Option("--mode", help="Filter by routing mode."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_connections.sync_detailed,
            columns=_COLS,
            max_widths=_MAXW,
            all_=all_,
            limit=limit,
            after=after,
            connector=connector,
            session_id=session_id,
            mode=mode,
        )

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        call_single(ctx, get_connection.sync_detailed, connection_id=connection_id)

    run_or_die(_run)


def _parse_secret_kvs(values: list[str]) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` ``--secret`` flags into a dict.

    Empty list returns an empty dict; the caller distinguishes that
    case from "no flag passed at all" before sending to the API.
    """
    out: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise PayloadError(f"--secret expects KEY=VALUE; got {raw!r}")
        key, _, val = raw.partition("=")
        if not key:
            raise PayloadError(f"--secret KEY must be non-empty; got {raw!r}")
        out[key] = val
    return out


@app.command("create", help="Create a connection in detached mode.")
def create(
    ctx: typer.Context,
    connector: Annotated[
        str | None, typer.Option("--connector", help="Connector type (e.g. signal).")
    ] = None,
    external_account_id: Annotated[
        str | None,
        typer.Option(
            "--external-account-id",
            help="External messaging identity (e.g. Signal phone, Telegram bot id).",
        ),
    ] = None,
    metadata_json: Annotated[
        str | None,
        typer.Option("--metadata-json", help="JSON object of connection metadata."),
    ] = None,
    secret: Annotated[
        list[str] | None,
        typer.Option(
            "--secret",
            help=(
                "Platform credential as KEY=VALUE (e.g. bot_token=12345:abc). "
                "Repeatable.  Encrypted at rest server-side; only ever read "
                "back by the connector container's own bearer token."
            ),
        ),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        ergonomic: dict[str, Any] | None = None
        if connector is not None or external_account_id is not None or secret:
            if connector is None or external_account_id is None:
                print_error("--connector and --external-account-id are both required")
                return 64
            ergonomic = {"connector": connector, "external_account_id": external_account_id}
            if metadata_json is not None:
                ergonomic["metadata"] = load_json_object(metadata_json, "--metadata-json")
            if secret:
                ergonomic["secrets"] = _parse_secret_kvs(secret)
        payload = resolve_payload(ergonomic, file, stdin, data)
        body = ConnectionCreate.from_dict(payload)
        call_single(ctx, create_connection.sync_detailed, body=body)
        return None

    run_or_die(_run)


@app.command(
    "set-secrets",
    help="Replace the connection's encrypted secrets dict (wholesale).",
)
def set_secrets(
    ctx: typer.Context,
    connection_id: str,
    secret: Annotated[
        list[str] | None,
        typer.Option(
            "--secret",
            help=(
                "Platform credential as KEY=VALUE (e.g. bot_token=12345:abc). "
                "Repeatable.  Encrypted at rest server-side; only ever read "
                "back by the connector container's own bearer token.  Pass "
                "no flags to clear all secrets on this connection."
            ),
        ),
    ] = None,
) -> None:
    def _run() -> int | None:
        kvs = _parse_secret_kvs(secret or [])
        body = ConnectionSetSecrets(secrets=ConnectionSetSecretsSecrets.from_dict(kvs))
        call_single(
            ctx, set_connection_secrets.sync_detailed, connection_id=connection_id, body=body
        )
        print_success("secrets updated on", connection_id)
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
        call_single(ctx, attach_connection.sync_detailed, connection_id=connection_id, body=body)

    run_or_die(_run)


@app.command("detach", help="Drop the single_session binding, leaving the connection detached.")
def detach(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        call_single(ctx, detach_connection.sync_detailed, connection_id=connection_id)

    run_or_die(_run)


@app.command("configure-per-chat", help="Switch the connection into per_chat mode.")
def configure_per_chat(
    ctx: typer.Context,
    connection_id: str,
    template: Annotated[str, typer.Option("--template", help="Session template id.")],
) -> None:
    def _run() -> None:
        body = ConnectionConfigurePerChat(session_template_id=template)
        call_single(
            ctx, configure_connection_per_chat.sync_detailed, connection_id=connection_id, body=body
        )

    run_or_die(_run)


@app.command(
    "unconfigure", help="Drop the per_chat configuration, leaving the connection detached."
)
def unconfigure(ctx: typer.Context, connection_id: str) -> None:
    def _run() -> None:
        call_single(ctx, unconfigure_connection.sync_detailed, connection_id=connection_id)

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
        call_single(ctx, bind_chat.sync_detailed, connection_id=connection_id, body=body)

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
