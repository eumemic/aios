"""@covers decorator + registry for the CLI coverage drift guard.

Each typer command annotates which OpenAPI ``operationId`` it implements
by stacking ``@covers("<operation_id>")`` underneath ``@app.command(...)``.
The drift test in :mod:`tests.unit.test_cli_coverage` compares the union of
``REGISTRY`` keys against the operations published in ``openapi.json`` and
the entries in :mod:`aios.cli.allowlist`, failing if any operation is
missing from both.

Stacking multiple ``@covers(...)`` on one command is allowed for the
genuinely-multi-operation case (e.g. ``aios status`` probes ``get_health``
and ``list_agents`` in the same body).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., object])

REGISTRY: dict[str, list[str]] = {}
"""Maps operationId → list of fully-qualified command function paths.

The value is a list (not a single string) so the same operation can
legitimately be claimed by more than one command — e.g. a top-level
alias plus the canonical subcommand. The test only cares that the key
exists, but keeping the call sites lets future tooling render a
"which command covers what" map.
"""


def covers(operation_id: str) -> Callable[[F], F]:
    """Annotate a typer command as implementing ``operation_id``.

    Usage::

        @app.command("list")
        @covers("list_agents")
        def list_(...) -> None: ...

    The decorator returns the function unchanged; it only records the
    mapping at import time so the coverage test can read it.
    """

    def decorator(fn: F) -> F:
        path = f"{fn.__module__}.{fn.__qualname__}"
        REGISTRY.setdefault(operation_id, []).append(path)
        return fn

    return decorator
