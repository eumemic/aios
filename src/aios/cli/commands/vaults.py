"""``aios vaults ...`` — vault CRUD + nested credentials subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import call_single, render_paginated, unwrap
from aios.cli.files import load_json_object, load_payload, resolve_payload
from aios.cli.output import print_error, print_success
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.vaults import (
    archive_vault,
    archive_vault_credential,
    create_vault,
    create_vault_credential,
    delete_vault,
    delete_vault_credential,
    get_vault,
    get_vault_credential,
    list_vault_credentials,
    list_vaults,
    update_vault,
    update_vault_credential,
)
from aios_sdk._generated.models.vault_create import VaultCreate
from aios_sdk._generated.models.vault_credential_create import VaultCredentialCreate
from aios_sdk._generated.models.vault_credential_update import VaultCredentialUpdate
from aios_sdk._generated.models.vault_update import VaultUpdate

app = typer.Typer(name="vaults", help="Manage vaults and credentials.", no_args_is_help=True)
credentials = typer.Typer(
    name="credentials",
    help="Manage credentials within a vault.",
    no_args_is_help=True,
)
app.add_typer(credentials, name="credentials")


# ── vaults CRUD ─────────────────────────────────────────────────────────────

_VAULT_COLS = ("id", "display_name", "archived_at", "updated_at")
_CRED_COLS = ("id", "display_name", "auth_type", "target_url", "updated_at")


@app.command("list", help="List vaults.")
def list_(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[str | None, typer.Option("--after")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_vaults.sync_detailed,
            columns=_VAULT_COLS,
            all_=all_,
            limit=limit,
            after=after,
        )

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, vault_id: str) -> None:
    def _run() -> None:
        call_single(ctx, get_vault.sync_detailed, vault_id=vault_id)

    run_or_die(_run)


@app.command("create", help="Create a vault.")
def create(
    ctx: typer.Context,
    display_name: Annotated[
        str | None,
        typer.Option("--display-name", help="Human-readable name (1-128 chars)."),
    ] = None,
    metadata_json: Annotated[
        str | None,
        typer.Option("--metadata-json", help="JSON object of vault metadata."),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        ergonomic: dict[str, Any] | None = None
        if display_name is not None or metadata_json is not None:
            if display_name is None:
                print_error("--display-name is required")
                return 64
            ergonomic = {"display_name": display_name}
            if metadata_json is not None:
                ergonomic["metadata"] = load_json_object(metadata_json, "--metadata-json")
        payload = resolve_payload(ergonomic, file, stdin, data)
        body = VaultCreate.from_dict(payload)
        call_single(ctx, create_vault.sync_detailed, body=body)
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
        payload = load_payload(file, stdin, data)
        body = VaultUpdate.from_dict(payload)
        call_single(ctx, update_vault.sync_detailed, vault_id=vault_id, body=body)
        return None

    run_or_die(_run)


@app.command("archive")
def archive(ctx: typer.Context, vault_id: str) -> None:
    def _run() -> None:
        call_single(ctx, archive_vault.sync_detailed, vault_id=vault_id)

    run_or_die(_run)


@app.command(
    "delete",
    help="Hard-delete a vault (irreversible; prefer `archive` instead).",
)
def delete(
    ctx: typer.Context,
    vault_id: str,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Required to confirm hard-delete (no interactive prompt)."),
    ] = False,
) -> None:
    def _run() -> int | None:
        if not yes:
            print_error(
                "hard-delete is irreversible; pass --yes to confirm "
                "(or use `aios vaults archive` for a reversible soft-delete)"
            )
            return 2
        with get_state(ctx).sdk_client() as client:
            unwrap(delete_vault.sync_detailed(client=client, vault_id=vault_id))
        print_success("deleted", vault_id)
        return None

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
        render_paginated(
            ctx,
            list_vault_credentials.sync_detailed,
            columns=_CRED_COLS,
            all_=all_,
            limit=limit,
            after=after,
            vault_id=vault_id,
        )

    run_or_die(_run)


@credentials.command("get")
def cred_get(ctx: typer.Context, vault_id: str, credential_id: str) -> None:
    def _run() -> None:
        call_single(
            ctx, get_vault_credential.sync_detailed, vault_id=vault_id, credential_id=credential_id
        )

    run_or_die(_run)


@credentials.command("create", help="Create a credential in a vault.")
def cred_create(
    ctx: typer.Context,
    vault_id: str,
    body_file: Annotated[
        Path | None,
        typer.Option(
            "--body-file",
            help=(
                "Path to a JSON file containing the full VaultCredentialCreate body. "
                "Using a file avoids leaking secrets (tokens, client secrets) into "
                "shell history. Alias for --file."
            ),
        ),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        source = body_file or file
        payload = load_payload(source, stdin, data)
        body = VaultCredentialCreate.from_dict(payload)
        call_single(ctx, create_vault_credential.sync_detailed, vault_id=vault_id, body=body)
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
        payload = load_payload(file, stdin, data)
        body = VaultCredentialUpdate.from_dict(payload)
        call_single(
            ctx,
            update_vault_credential.sync_detailed,
            vault_id=vault_id,
            credential_id=credential_id,
            body=body,
        )
        return None

    run_or_die(_run)


@credentials.command("archive")
def cred_archive(ctx: typer.Context, vault_id: str, credential_id: str) -> None:
    def _run() -> None:
        call_single(
            ctx,
            archive_vault_credential.sync_detailed,
            vault_id=vault_id,
            credential_id=credential_id,
        )

    run_or_die(_run)


@credentials.command(
    "delete",
    help="Hard-delete a credential (irreversible; prefer `archive` instead).",
)
def cred_delete(
    ctx: typer.Context,
    vault_id: str,
    credential_id: str,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Required to confirm hard-delete (no interactive prompt)."),
    ] = False,
) -> None:
    def _run() -> int | None:
        if not yes:
            print_error(
                "hard-delete is irreversible; pass --yes to confirm "
                "(or use `aios vaults credentials archive` for a reversible soft-delete)"
            )
            return 2
        with get_state(ctx).sdk_client() as client:
            unwrap(
                delete_vault_credential.sync_detailed(
                    client=client, vault_id=vault_id, credential_id=credential_id
                )
            )
        print_success("deleted", credential_id)
        return None

    run_or_die(_run)
