"""``aios connector-tokens ...`` — issue, list, and revoke per-connection
bearer tokens (#301).

The plaintext token is printed to stdout ONCE on issue.  Pipe it to a
secret store immediately — the CLI doesn't persist it and the API
won't surface it again.
"""

from __future__ import annotations

from typing import Annotated

import typer

from aios.cli.commands._shared import render_list, render_single, unwrap
from aios.cli.runtime import get_state, run_or_die
from aios.sdk._generated.api.connector_tokens import (
    issue_connector_token,
    list_connector_tokens,
    revoke_connector_token,
)
from aios.sdk._generated.models.connector_token_issue import ConnectorTokenIssue

app = typer.Typer(
    name="connector-tokens", help="Manage connector bearer tokens.", no_args_is_help=True
)

_COLS = ("id", "connection_id", "label", "created_at", "last_used_at", "revoked_at")
_MAXW = {"connection_id": 28, "label": 24}


@app.command("issue", help="Mint a new bearer token for a connection.")
def issue(
    ctx: typer.Context,
    connection_id: Annotated[str, typer.Option("--connection-id")],
    label: Annotated[str | None, typer.Option("--label")] = None,
) -> None:
    """Issue a fresh token. The plaintext is printed ONCE — save it now."""

    def _run() -> None:
        body = ConnectorTokenIssue(connection_id=connection_id, label=label)
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(issue_connector_token.sync_detailed(client=client, body=body))
        render_single(obj.to_dict())

    run_or_die(_run)


@app.command("list")
def list_(
    ctx: typer.Context,
    connection_id: Annotated[str, typer.Option("--connection-id")],
) -> None:
    """List all tokens for a connection (revoked included)."""

    def _run() -> None:
        state = get_state(ctx)
        with state.sdk_client() as client:
            obj = unwrap(
                list_connector_tokens.sync_detailed(client=client, connection_id=connection_id)
            )
        render_list(
            state.output_format,
            obj.to_dict(),
            columns=_COLS,
            max_widths=_MAXW,
        )

    run_or_die(_run)


@app.command("revoke", help="Soft-delete a token. Idempotent.")
def revoke(ctx: typer.Context, token_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(revoke_connector_token.sync_detailed(client=client, token_id=token_id))
        render_single(obj.to_dict())

    run_or_die(_run)
