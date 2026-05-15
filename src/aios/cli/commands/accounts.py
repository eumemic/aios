"""``aios accounts ...`` — multi-tenancy management plane (#367 PR 8).

Surfaces the eight management endpoints from PR 7:
    aios accounts me
    aios accounts list                          # direct children
    aios accounts get <ID>                      # caller-or-child read
    aios accounts mint --display-name N [...]   # mint child + first key
    aios accounts archive <ID>                  # archive direct child

    aios accounts keys list <ID>
    aios accounts keys mint <ID> --label L
    aios accounts keys revoke <ID> <KEY_ID>

The newly-minted plaintext bearers are printed exactly once — they're
unrecoverable after the response, so we print them on a separate line
prefixed ``plaintext_key:`` to make it obvious what to copy and so a
``--format json`` consumer can extract via the key.
"""

from __future__ import annotations

from typing import Annotated

import typer

from aios.cli.commands._shared import render_list, render_single, unwrap
from aios.cli.output import print_json, print_note
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.accounts import (
    archive_account,
    get_account,
    get_my_account,
    list_account_keys,
    list_my_children,
    mint_account_key,
    mint_child_account,
    revoke_account_key,
)
from aios_sdk._generated.models.mint_account_request import MintAccountRequest
from aios_sdk._generated.models.mint_key_request import MintKeyRequest

app = typer.Typer(
    name="accounts",
    help="Manage child accounts and API keys for the caller's tenant.",
    no_args_is_help=True,
)


_ACCOUNT_COLS = ("id", "display_name", "can_mint_children", "archived_at", "created_at")
_KEY_COLS = ("key_id", "label", "created_at", "revoked_at")


def _envelope[T](items: list[T]) -> dict[str, object]:
    """Wrap a bare list response from the management endpoints into the
    envelope shape :func:`render_list` expects.

    Management list endpoints return ``list[Account]`` directly (not the
    paginated ``ListResponse`` shape resource lists use), but the table
    renderer is paginated-envelope-shaped so we adapt at the edge.
    """
    return {
        "data": [item.to_dict() for item in items],  # type: ignore[attr-defined]
        "has_more": False,
        "next_after": None,
    }


@app.command("me")
def me(ctx: typer.Context) -> None:
    """Print the caller's account row."""

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(get_my_account.sync_detailed(client=client))
        render_single(obj.to_dict())

    run_or_die(_run)


@app.command("list")
def list_(ctx: typer.Context) -> None:
    """List direct, non-archived child accounts under the caller."""

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            resp = unwrap(list_my_children.sync_detailed(client=client))
        state = get_state(ctx)
        render_list(state.output_format, _envelope(resp), columns=_ACCOUNT_COLS)

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, target_id: str) -> None:
    """Read a caller-or-direct-child account."""

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(get_account.sync_detailed(client=client, target_id=target_id))
        render_single(obj.to_dict())

    run_or_die(_run)


@app.command("mint", help="Mint a direct child account and its first API key.")
def mint(
    ctx: typer.Context,
    display_name: Annotated[
        str,
        typer.Option("--display-name", "-n", help="Human-readable name for the child."),
    ],
    can_mint_children: Annotated[
        bool,
        typer.Option(
            "--can-mint-children/--no-can-mint-children",
            help="Whether the new child can mint grandchildren.",
        ),
    ] = False,
) -> None:
    def _run() -> None:
        body = MintAccountRequest(display_name=display_name, can_mint_children=can_mint_children)
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(mint_child_account.sync_detailed(client=client, body=body))
        data = obj.to_dict()
        if get_state(ctx).output_format == "json":
            print_json(data)
        else:
            render_single({"account_id": data["account_id"], "key_id": data["key_id"]})
            # Plaintext key is unrecoverable after this response; show it
            # on its own line so it's hard to miss. Goes to stderr so a
            # ``--format json`` consumer's pipe stays clean.
            print_note(f"plaintext_key: {data['plaintext_key']}")

    run_or_die(_run)


@app.command("archive", help="Archive a direct child account.")
def archive(ctx: typer.Context, target_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(archive_account.sync_detailed(client=client, target_id=target_id))
        render_single(obj.to_dict())

    run_or_die(_run)


keys_app = typer.Typer(
    name="keys",
    help="Manage API keys on the caller's account or its direct children.",
    no_args_is_help=True,
)


@keys_app.command("list")
def keys_list(ctx: typer.Context, target_id: str) -> None:
    """List key summaries for ``target_id`` (caller or direct child)."""

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            resp = unwrap(list_account_keys.sync_detailed(client=client, target_id=target_id))
        state = get_state(ctx)
        render_list(state.output_format, _envelope(resp), columns=_KEY_COLS)

    run_or_die(_run)


@keys_app.command("mint", help="Mint an additional API key on ``target_id``.")
def keys_mint(
    ctx: typer.Context,
    target_id: str,
    label: Annotated[str, typer.Option("--label", "-l", help="Operator-visible key label.")],
) -> None:
    def _run() -> None:
        body = MintKeyRequest(label=label)
        with get_state(ctx).sdk_client() as client:
            obj = unwrap(
                mint_account_key.sync_detailed(client=client, target_id=target_id, body=body)
            )
        data = obj.to_dict()
        if get_state(ctx).output_format == "json":
            print_json(data)
        else:
            render_single({"key_id": data["key_id"]})
            print_note(f"plaintext_key: {data['plaintext_key']}")

    run_or_die(_run)


@keys_app.command("revoke")
def keys_revoke(ctx: typer.Context, target_id: str, key_id: str) -> None:
    """Revoke a key. Idempotent — re-revoking is a no-op."""

    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            unwrap(
                revoke_account_key.sync_detailed(client=client, target_id=target_id, key_id=key_id)
            )
        print_note(f"revoked {key_id} on {target_id}")

    run_or_die(_run)


app.add_typer(keys_app, name="keys")
