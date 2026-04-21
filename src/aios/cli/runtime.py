"""Per-invocation CLI state + the error-handling shim used by every command.

Split out of :mod:`aios.cli.app` so command modules can import it without
introducing a circular dependency (the root app imports every command
module at its tail to register sub-apps).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import typer

from aios.cli.client import AiosApiError, AiosClient
from aios.cli.output import OutputFormat, print_error


@dataclass(slots=True)
class CliState:
    """Per-invocation CLI state set from the root callback."""

    base_url: str
    api_key: str | None
    output_format: OutputFormat
    verbose: bool

    def client(self) -> AiosClient:
        return AiosClient(base_url=self.base_url, api_key=self.api_key)


def get_state(ctx: typer.Context) -> CliState:
    """Return the :class:`CliState` attached to the typer context."""
    state = ctx.obj
    assert isinstance(state, CliState)
    return state


def run_or_die(fn: Callable[[], int | None]) -> None:
    """Execute ``fn`` and translate ``AiosApiError`` into a clean CLI exit.

    Keeps tracebacks hidden and prints the server's error envelope.
    Non-API exceptions bubble.
    """
    try:
        rc = fn()
    except AiosApiError as exc:
        print_error(f"{exc.error_type}: {exc.message}")
        if exc.detail:
            print_error(f"detail: {exc.detail}")
        exit_code = 2 if exc.status_code == 401 else 1
        raise typer.Exit(exit_code) from exc
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        raise typer.Exit(130) from None
    if rc:
        raise typer.Exit(rc)
