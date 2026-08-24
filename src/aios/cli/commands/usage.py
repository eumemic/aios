"""``aios usage ...`` — live inference attribution views."""

from __future__ import annotations

from typing import Annotated

import typer

from aios.cli.commands._shared import call_single
from aios.cli.coverage import covers
from aios.cli.runtime import run_or_die
from aios.models.accounting import (
    DEFAULT_USAGE_WINDOW_SECONDS,
    MAX_USAGE_WINDOW_SECONDS,
    MIN_USAGE_WINDOW_SECONDS,
)
from aios_sdk._generated.api.usage import list_usage_consumers
from aios_sdk._generated.models.list_usage_consumers_metric import ListUsageConsumersMetric

app = typer.Typer(
    name="usage",
    help="Inspect live inference attribution and rates.",
    no_args_is_help=True,
)


@app.command("consumers")
@covers("list_usage_consumers")
def consumers(
    ctx: typer.Context,
    window_seconds: Annotated[
        int,
        typer.Option(
            "--window-seconds",
            min=MIN_USAGE_WINDOW_SECONDS,
            max=MAX_USAGE_WINDOW_SECONDS,
            help="Rolling window in seconds (60 seconds through 30 days).",
        ),
    ] = DEFAULT_USAGE_WINDOW_SECONDS,
    metric: Annotated[
        ListUsageConsumersMetric,
        typer.Option("--metric", help="Rank by subtree cost or total-token rate."),
    ] = ListUsageConsumersMetric.COST_MICROUSD,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100, help="Maximum root consumers to return."),
    ] = 20,
) -> None:
    """Rank additive account roots by live creation-subtree inference rate."""

    def _run() -> None:
        call_single(
            ctx,
            list_usage_consumers.sync_detailed,
            window_seconds=window_seconds,
            metric=metric,
            limit=limit,
        )

    run_or_die(_run)
