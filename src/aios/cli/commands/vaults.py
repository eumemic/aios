"""``aios vaults ...`` — vault CRUD + nested credentials subcommands."""

from __future__ import annotations

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
from aios.cli.files import PayloadError, load_json_object, load_payload, resolve_payload
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
        render_list(state.output_format, envelope, columns=_VAULT_COLS)

    run_or_die(_run)


@app.command("get")
def get(ctx: typer.Context, vault_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/vaults/{vault_id}")
        render_single(obj)

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
        client = just_client(ctx)
        with client:
            obj = client.request("POST", "/v1/vaults", json_body=payload)
        render_single(obj)
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
        client = just_client(ctx)
        with client:
            obj = client.request("PUT", f"/v1/vaults/{vault_id}", json_body=payload)
        render_single(obj)
        return None

    run_or_die(_run)


@app.command("archive")
def archive(ctx: typer.Context, vault_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("POST", f"/v1/vaults/{vault_id}/archive")
        render_single(obj)

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
            print_error("hard-delete is irreversible; pass --yes to confirm")
            return 2
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/vaults/{vault_id}")
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
        state, client = with_client(ctx)
        path = f"/v1/vaults/{vault_id}/credentials"
        with client:
            envelope = (
                fetch_all(client, path)
                if all_
                else client.request("GET", path, params={"limit": limit, "after": after})
            )
        render_list(state.output_format, envelope, columns=_CRED_COLS)

    run_or_die(_run)


@credentials.command("get")
def cred_get(ctx: typer.Context, vault_id: str, credential_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request("GET", f"/v1/vaults/{vault_id}/credentials/{credential_id}")
        render_single(obj)

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
        try:
            payload = load_payload(source, stdin, data)
        except PayloadError as exc:
            print_error(str(exc))
            return 64
        client = just_client(ctx)
        with client:
            obj = client.request("POST", f"/v1/vaults/{vault_id}/credentials", json_body=payload)
        render_single(obj)
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
        client = just_client(ctx)
        with client:
            obj = client.request(
                "PUT",
                f"/v1/vaults/{vault_id}/credentials/{credential_id}",
                json_body=payload,
            )
        render_single(obj)
        return None

    run_or_die(_run)


@credentials.command("archive")
def cred_archive(ctx: typer.Context, vault_id: str, credential_id: str) -> None:
    def _run() -> None:
        client = just_client(ctx)
        with client:
            obj = client.request(
                "POST", f"/v1/vaults/{vault_id}/credentials/{credential_id}/archive"
            )
        render_single(obj)

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
            print_error("hard-delete is irreversible; pass --yes to confirm")
            return 2
        client = just_client(ctx)
        with client:
            client.request("DELETE", f"/v1/vaults/{vault_id}/credentials/{credential_id}")
        return None

    run_or_die(_run)
