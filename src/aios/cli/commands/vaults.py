"""``aios vaults ...`` — vault CRUD + nested credentials subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.commands._shared import (
    fetch_all,
    render_list,
    render_single,
    with_client,
)
from aios.cli.files import PayloadError, load_payload
from aios.cli.output import print_error
from aios.cli.runtime import run_or_die

app = typer.Typer(name="vaults", help="Manage vaults and credentials.", no_args_is_help=True)
credentials = typer.Typer(
    name="credentials",
    help="Manage credentials within a vault.",
    no_args_is_help=True,
)
app.add_typer(credentials, name="credentials")


# ── vaults CRUD ─────────────────────────────────────────────────────────────

_VAULT_COLS = ("id", "display_name", "archived_at", "updated_at")
_CRED_COLS = ("id", "display_name", "auth_type", "mcp_server_url", "updated_at")


@app.command("list", help="List vaults.")
def list_(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        with client:
            envelope = (
                fetch_all(client, "/v1/vaults")
                if all_
                else client.request("GET", "/v1/vaults", params={"limit": limit, "after": after})
            )
        render_list(state, envelope, columns=_VAULT_COLS)

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, vault_id: str) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/vaults/{vault_id}")
        render_single(state, obj)

    run_or_die(_run)


@app.command("create", help="Create a vault (VaultCreate shape).")
def create(
    ctx: typer.Context,
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
        state, client = with_client(ctx)
        with client:
            obj = client.request("POST", "/v1/vaults", json_body=payload)
        render_single(state, obj)
        return None

    run_or_die(_run)


@app.command("update", help="Update a vault (VaultUpdate shape).")
def update(
    ctx: typer.Context,
    vault_id: str,
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
        state, client = with_client(ctx)
        with client:
            obj = client.request("PUT", f"/v1/vaults/{vault_id}", json_body=payload)
        render_single(state, obj)
        return None

    run_or_die(_run)


@app.command("archive")
def archive(ctx: typer.Context, vault_id: str) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        with client:
            obj = client.request("POST", f"/v1/vaults/{vault_id}/archive")
        render_single(state, obj)

    run_or_die(_run)


@app.command("delete")
def delete(ctx: typer.Context, vault_id: str) -> None:
    def _run() -> None:
        _state, client = with_client(ctx)
        with client:
            client.request("DELETE", f"/v1/vaults/{vault_id}")

    run_or_die(_run)


# ── credentials ────────────────────────────────────────────────────────────


@credentials.command("list")
def cred_list(
    ctx: typer.Context,
    vault_id: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        path = f"/v1/vaults/{vault_id}/credentials"
        with client:
            envelope = (
                fetch_all(client, path)
                if all_
                else client.request("GET", path, params={"limit": limit, "after": after})
            )
        render_list(state, envelope, columns=_CRED_COLS)

    run_or_die(_run)


@credentials.command("get")
def cred_get(ctx: typer.Context, vault_id: str, credential_id: str) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/vaults/{vault_id}/credentials/{credential_id}")
        render_single(state, obj)

    run_or_die(_run)


@credentials.command("create", help="Create a credential in a vault (VaultCredentialCreate).")
def cred_create(
    ctx: typer.Context,
    vault_id: str,
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
        state, client = with_client(ctx)
        with client:
            obj = client.request("POST", f"/v1/vaults/{vault_id}/credentials", json_body=payload)
        render_single(state, obj)
        return None

    run_or_die(_run)


@credentials.command("update")
def cred_update(
    ctx: typer.Context,
    vault_id: str,
    credential_id: str,
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
        state, client = with_client(ctx)
        with client:
            obj = client.request(
                "PUT",
                f"/v1/vaults/{vault_id}/credentials/{credential_id}",
                json_body=payload,
            )
        render_single(state, obj)
        return None

    run_or_die(_run)


@credentials.command("archive")
def cred_archive(ctx: typer.Context, vault_id: str, credential_id: str) -> None:
    def _run() -> None:
        state, client = with_client(ctx)
        with client:
            obj = client.request(
                "POST", f"/v1/vaults/{vault_id}/credentials/{credential_id}/archive"
            )
        render_single(state, obj)

    run_or_die(_run)


@credentials.command("delete")
def cred_delete(ctx: typer.Context, vault_id: str, credential_id: str) -> None:
    def _run() -> None:
        _state, client = with_client(ctx)
        with client:
            client.request("DELETE", f"/v1/vaults/{vault_id}/credentials/{credential_id}")

    run_or_die(_run)
