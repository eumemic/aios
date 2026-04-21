"""Root typer app for the aios CLI.

Wires every subcommand module and exposes the global options (``--url``,
``--api-key``, ``--format``, ``--verbose``) via a callback. Commands read
the resolved state from :class:`typer.Context` via :func:`get_state` in
:mod:`aios.cli.runtime`.
"""

from __future__ import annotations

from typing import Annotated

import typer

from aios.cli.config import resolve_base_url
from aios.cli.output import OutputFormat
from aios.cli.runtime import CliState

app = typer.Typer(
    name="aios",
    help=(
        "aios command-line interface. Manages agents, sessions, and all other "
        "resources against a running aios API. The first positional dictates the "
        "subcommand (e.g. `aios chat`, `aios agents list`, `aios sessions stream`)."
    ),
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_enable=False,
)


@app.callback()
def _root(
    ctx: typer.Context,
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            envvar="AIOS_URL",
            help="API base URL. Defaults to AIOS_URL env or http://127.0.0.1:{AIOS_API_PORT}.",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            envvar="AIOS_API_KEY",
            help="Bearer token for the API. Required for protected endpoints.",
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help="Output format. 'table' for lists, 'json' for machine-readable output.",
            case_sensitive=False,
        ),
    ] = "table",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print additional context (full tool output, etc)."),
    ] = False,
) -> None:
    base_url = resolve_base_url(url)
    ctx.obj = CliState(
        base_url=base_url,
        api_key=api_key,
        output_format=output_format,
        verbose=verbose,
    )


# Subcommand registration. Imports live here (at the bottom of the module)
# so command modules can import from ``aios.cli.runtime`` / ``aios.cli.app``
# without causing a circular import.
from aios.cli.commands import agents as _agents  # noqa: E402
from aios.cli.commands import bindings as _bindings  # noqa: E402
from aios.cli.commands import chat as _chat  # noqa: E402
from aios.cli.commands import connections as _connections  # noqa: E402
from aios.cli.commands import envs as _envs  # noqa: E402
from aios.cli.commands import ops as _ops  # noqa: E402
from aios.cli.commands import rules as _rules  # noqa: E402
from aios.cli.commands import sessions as _sessions  # noqa: E402
from aios.cli.commands import skills as _skills  # noqa: E402
from aios.cli.commands import status as _status  # noqa: E402
from aios.cli.commands import tail as _tail  # noqa: E402
from aios.cli.commands import vaults as _vaults  # noqa: E402

app.add_typer(_agents.app, name="agents")
app.add_typer(_sessions.app, name="sessions")
app.add_typer(_skills.app, name="skills")
app.add_typer(_vaults.app, name="vaults")
app.add_typer(_connections.app, name="connections")
app.add_typer(_bindings.app, name="bindings")
app.add_typer(_rules.app, name="rules")
app.add_typer(_envs.app, name="envs")

_ops.register(app)
_status.register(app)
_chat.register(app)
_tail.register(app)
