"""Drift guard: every OpenAPI operation has a CLI command or is allowlisted.

The aios CLI is hand-written, so a new FastAPI route can land without an
``aios <verb>`` command unless something forces the issue at PR time.
This test is that forcing function.

Three invariants:

1. Every ``operationId`` in :mod:`openapi.json` must be either claimed by
   an ``@covers(...)`` decorator on a CLI command, or listed in
   :mod:`aios.cli.allowlist`. Otherwise: drift.

2. Every entry in the allowlist must correspond to a real operation in
   :mod:`openapi.json` — dead entries are stale rot.

3. The two sources are mutually exclusive: an operation that has a CLI
   command must not also be allowlisted (or the comment is misleading
   about why coverage is absent).

A new route landing without a CLI command surfaces here as a friendly
``"operation 'X' has no CLI coverage and is not allowlisted"`` — author
must either add a typer command + ``@covers("X")`` or explicitly
allowlist X in :mod:`aios.cli.allowlist`.
"""

from __future__ import annotations

import json
from pathlib import Path

# Importing the root typer app transitively imports every command module,
# which fires the @covers(...) decorators and populates REGISTRY.
import aios.cli.app  # noqa: F401
from aios.cli.allowlist import NEEDS_CLI_TRACKED, NOT_CLI_OPERATIONS, all_allowlisted
from aios.cli.coverage import REGISTRY


def _openapi_operations() -> set[str]:
    """Read every operationId from the committed openapi.json."""
    openapi_path = Path(__file__).resolve().parents[2] / "openapi.json"
    spec = json.loads(openapi_path.read_text())
    ops: set[str] = set()
    for methods in spec["paths"].values():
        for info in methods.values():
            op_id = info.get("operationId")
            assert op_id, f"openapi.json operation missing operationId: {info}"
            ops.add(op_id)
    return ops


def test_every_operation_is_covered_or_allowlisted() -> None:
    """Each operation has a CLI command (via @covers) or is allowlisted."""
    operations = _openapi_operations()
    covered = set(REGISTRY)
    allowlisted = all_allowlisted()

    missing = sorted(operations - covered - allowlisted)
    assert not missing, (
        f"{len(missing)} OpenAPI operation(s) have no CLI coverage and are not "
        f"allowlisted:\n  "
        + "\n  ".join(missing)
        + "\n\nEither:\n"
        + "  • add a typer command in src/aios/cli/commands/ and decorate it with "
        + '@covers("<operation_id>")\n'
        + "  • or add an entry to src/aios/cli/allowlist.py explaining why no CLI is "
        + "appropriate (NOT_CLI_OPERATIONS) or pointing at a tracking issue "
        + "(NEEDS_CLI_TRACKED)."
    )


def test_allowlist_entries_correspond_to_real_operations() -> None:
    """Allowlist rot check: no dead entries."""
    operations = _openapi_operations()

    dead_not_cli = sorted(set(NOT_CLI_OPERATIONS) - operations)
    dead_tracked = sorted(set(NEEDS_CLI_TRACKED) - operations)
    assert not dead_not_cli, (
        "NOT_CLI_OPERATIONS contains entries that no longer exist in openapi.json — "
        "remove or update them:\n  " + "\n  ".join(dead_not_cli)
    )
    assert not dead_tracked, (
        "NEEDS_CLI_TRACKED contains entries that no longer exist in openapi.json — "
        "remove or update them:\n  " + "\n  ".join(dead_tracked)
    )


def test_no_operation_is_both_covered_and_allowlisted() -> None:
    """Consistency: a covered operation must not also be allowlisted."""
    double = sorted(set(REGISTRY) & all_allowlisted())
    assert not double, (
        "These operations are both @covers'd by a CLI command AND in the allowlist "
        "— remove the allowlist entry (the @covers wins):\n  " + "\n  ".join(double)
    )


def test_allowlist_categories_are_disjoint() -> None:
    """NOT_CLI_OPERATIONS and NEEDS_CLI_TRACKED must not overlap."""
    overlap = sorted(set(NOT_CLI_OPERATIONS) & set(NEEDS_CLI_TRACKED))
    assert not overlap, (
        "Allowlist categories overlap — an operation is either deliberately not for "
        "CLI or waiting on a CLI followup, not both:\n  " + "\n  ".join(overlap)
    )


def test_needs_cli_tracked_cite_real_issues() -> None:
    """Every deferred-CLI entry cites a real filed issue, never ``aios#TBD``.

    ``aios#TBD`` meant "tracking issue filed alongside this allowlist" — but those
    issues (the followups #370 promised) went unfiled, so the deferrals were
    untracked in practice. Filing them and asserting the placeholder is gone makes
    a deferred CLI command unable to ship untracked again (#1433 drift guard). We
    check dict *values* rather than scanning the source so this guard doesn't match
    its own needle (the docstring legitimately names ``aios#TBD``).
    """
    untracked = sorted(op for op, reason in NEEDS_CLI_TRACKED.items() if "aios#TBD" in reason)
    assert not untracked, (
        f"{len(untracked)} NEEDS_CLI_TRACKED entry(ies) cite the placeholder aios#TBD "
        "instead of a real filed issue — file a tracking issue and cite its #number:\n  "
        + "\n  ".join(untracked)
    )
