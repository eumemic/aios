"""``aios model-providers ...`` — per-account model-provider config CRUD.

``create``/``update`` take the full request body via ``--file``/``--stdin``/
``--data`` only (no plain ``--api-key VALUE`` flag) — same secret-hygiene
convention as ``aios vaults credentials create/update``: a provider API key
passed directly on argv lands in shell history and is visible to other
processes on the box via ``ps`` for the command's duration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.commands._shared import call_single, render_paginated, unwrap
from aios.cli.coverage import covers
from aios.cli.files import load_payload
from aios.cli.output import print_success
from aios.cli.runtime import get_state, run_or_die
from aios_sdk._generated.api.model_providers import (
    archive_model_provider,
    create_model_provider,
    get_model_provider,
    list_model_providers,
    update_model_provider,
)
from aios_sdk._generated.models.model_provider_create import ModelProviderCreate
from aios_sdk._generated.models.model_provider_update import ModelProviderUpdate

app = typer.Typer(
    name="model-providers",
    help="Manage per-account model-provider configs (API keys + proxy base URLs).",
    no_args_is_help=True,
)

_COLS = ("id", "provider", "api_base", "api_key_set", "archived_at")


@app.command("list")
@covers("list_model_providers")
def list_(
    ctx: typer.Context,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        render_paginated(
            ctx,
            list_model_providers.sync_detailed,
            columns=_COLS,
            all_=all_,
            limit=limit,
            provider=provider,
        )

    run_or_die(_run)


@app.command("get")
@covers("get_model_provider")
def get(ctx: typer.Context, model_provider_id: str) -> None:
    def _run() -> None:
        call_single(ctx, get_model_provider.sync_detailed, model_provider_id=model_provider_id)

    run_or_die(_run)


_BODY_FILE_HELP = (
    "Path to a JSON file containing the full {model} body. Using a file/stdin "
    "avoids leaking the provider api_key into shell history. Alias for --file."
)


@app.command(
    "create",
    help="Create a model-provider config for this account (ModelProviderCreate shape).",
)
@covers("create_model_provider")
def create(
    ctx: typer.Context,
    body_file: Annotated[
        Path | None,
        typer.Option("--body-file", help=_BODY_FILE_HELP.format(model="ModelProviderCreate")),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> None:
        payload = load_payload(body_file or file, stdin, data)
        body = ModelProviderCreate.from_dict(payload)
        call_single(ctx, create_model_provider.sync_detailed, body=body)

    run_or_die(_run)


@app.command(
    "update",
    help="Rotate the key and/or edit api_base (ModelProviderUpdate shape).",
)
@covers("update_model_provider")
def update(
    ctx: typer.Context,
    model_provider_id: str,
    body_file: Annotated[
        Path | None,
        typer.Option("--body-file", help=_BODY_FILE_HELP.format(model="ModelProviderUpdate")),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> None:
        payload = load_payload(body_file or file, stdin, data)
        body = ModelProviderUpdate.from_dict(payload)
        call_single(
            ctx, update_model_provider.sync_detailed, model_provider_id=model_provider_id, body=body
        )

    run_or_die(_run)


@app.command("archive", help="Archive a model-provider config (zeroes the encrypted key).")
@covers("archive_model_provider")
def archive(ctx: typer.Context, model_provider_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            unwrap(
                archive_model_provider.sync_detailed(
                    client=client, model_provider_id=model_provider_id
                )
            )
        print_success("archived", model_provider_id)

    run_or_die(_run)
