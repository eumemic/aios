"""@covers decorator + registry for the CLI coverage drift guard.

Each typer command annotates which OpenAPI ``operationId`` it implements
by stacking ``@covers("<operation_id>")`` underneath ``@app.command(...)``.
The drift test in :mod:`tests.unit.test_cli_coverage` compares ``REGISTRY``
against the operations published in ``openapi.json`` and the entries in
:mod:`aios.cli.allowlist`, failing if any operation is missing from both.

Stacking multiple ``@covers(...)`` on one command is allowed for the
genuinely-multi-operation case (e.g. ``aios status`` probes ``get_health``
and ``list_agents`` in the same body).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., object])

REGISTRY: set[str] = set()
"""The set of operationIds claimed by some ``@covers(...)`` CLI command.

A set, not a map: the drift test only needs membership ("is this operation
covered?"). Multiple commands may claim the same operation (a top-level alias
plus the canonical subcommand); they collapse to one entry, which is exactly
what the coverage check wants.
"""


def covers(operation_id: str) -> Callable[[F], F]:
    """Annotate a typer command as implementing ``operation_id``.

    Usage::

        @app.command("list")
        @covers("list_agents")
        def list_(...) -> None: ...

    The decorator returns the function unchanged; it only records the
    operation_id at import time so the coverage test can read it.
    """

    def decorator(fn: F) -> F:
        REGISTRY.add(operation_id)
        return fn

    return decorator
