"""``aios model-providers ...`` — per-account model-provider config CRUD."""

from __future__ import annotations

from typing import Annotated

import typer

from aios.cli.commands._shared import call_single, render_paginated, unwrap
from aios.cli.coverage import covers
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
from aios_sdk._generated.types import UNSET

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


@app.command("create", help="Create a model-provider config for this account.")
@covers("create_model_provider")
def create(
    ctx: typer.Context,
    provider: Annotated[
        str, typer.Option("--provider", help="LiteLLM provider name, e.g. anthropic.")
    ],
    api_key: Annotated[str, typer.Option("--api-key", help="Provider API key.")],
    api_base: Annotated[
        str | None, typer.Option("--api-base", help="Proxy/self-hosted base URL.")
    ] = None,
) -> None:
    def _run() -> None:
        body = ModelProviderCreate(
            provider=provider, api_key=api_key, api_base=api_base if api_base is not None else UNSET
        )
        call_single(ctx, create_model_provider.sync_detailed, body=body)

    run_or_die(_run)


@app.command("update", help="Rotate the key and/or edit api_base.")
@covers("update_model_provider")
def update(
    ctx: typer.Context,
    model_provider_id: str,
    api_key: Annotated[
        str | None, typer.Option("--api-key", help="New key. Omit to keep the existing one.")
    ] = None,
    api_base: Annotated[
        str | None, typer.Option("--api-base", help="New base URL. Omit to keep the existing one.")
    ] = None,
    clear_api_base: Annotated[
        bool, typer.Option("--clear-api-base", help="Clear api_base back to unset.")
    ] = False,
) -> None:
    def _run() -> None:
        body = ModelProviderUpdate(
            api_key=api_key if api_key is not None else UNSET,
            api_base=None if clear_api_base else (api_base if api_base is not None else UNSET),
        )
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
