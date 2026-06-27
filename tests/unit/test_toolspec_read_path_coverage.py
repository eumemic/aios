"""CI read-path coverage for persisted ``ToolSpec`` surfaces (#1577, epic #1572).

Two static-analysis checks that together close the "an unregistered raw-dict
surface is invisible to both the metric and the guard" blind spot, keeping the
per-site validator-resident tolerance intact:

**Part A — AST-lint (aimed at the RIGHT thing).**
:func:`aios.retirements.read_paths.iter_foreign_toolspec_parses` asserts a
persisted ``tools`` / ``tools_schema`` JSONB array element is **only ever parsed
into a** :class:`~aios.models.agents.ToolSpec` — never validated into some other
Pydantic model that would bypass the quarantine before-validator (and thus the
registry-driven read-tolerance + the boot scan's coverage). It is deliberately
**NOT** "ban ``model_validate`` outside a loader" — the per-site tolerance lives
at every ``ToolSpec.model_validate`` site and must stay. The build fails only on
a *new bypassing read path* that materialises a persisted tools array into a
model OTHER than ``ToolSpec``.

**Part B — Reflective surface-coverage test.**
:func:`aios.retirements.read_paths.iter_toolspec_consumed_columns` enumerates
every JSONB column **consumed as a ``ToolSpec``** — by the call-graph linkage of
SQL ``SELECT`` → ``ToolSpec`` consumer, NOT a hand-list — and this test FAILS if
any such column is absent from a descriptor's surface list. It keys on
"consumed-as-ToolSpec" not column type, which is exactly why it flags
``connectors.tools_schema`` (selected ``cat.tools_schema AS tools`` and fed to
``ToolSpec.model_validate`` through the connection tool-provider) as the seventh
surface — the silent hole the ad-hoc #1419 retirement missed.

These are pure ``ast`` + SQL-scan checks over the in-tree source — no DB, no
import side effects — so they run in the fast unit/lint tier (the same tier as
``test_append_event_structure``).
"""

from __future__ import annotations

import dataclasses

import pytest

import aios.retirements.registry as reg
from aios.retirements.read_paths import (
    ConsumedColumn,
    _columns_from_sql,
    iter_foreign_toolspec_parses,
    iter_toolspec_consumed_columns,
    surface_columns,
)

# The connector tools_schema the ad-hoc #1419 retirement missed — the seventh
# surface this whole epic is about.
SEVENTH_SURFACE = ("connectors", "tools_schema")


# ---------------------------------------------------------------------------
# Part A — AST-lint: persisted tools parse only into ToolSpec.
# ---------------------------------------------------------------------------


def test_no_read_path_parses_persisted_tools_into_a_foreign_model() -> None:
    """ACCEPTANCE: a read path parsing persisted tools into a non-ToolSpec model
    (a bare-dict / foreign-model bypass of the quarantine validator) FAILS CI.

    The current tree has none — every read site materialising a persisted tools
    array goes through ``ToolSpec`` (``load_tool_specs`` or
    ``ToolSpec.model_validate``). A regression that introduces a foreign model
    over a tools array would surface here.
    """

    offenders = list(iter_foreign_toolspec_parses())
    assert offenders == [], (
        "Persisted tools/tools_schema arrays must parse ONLY into ToolSpec, never "
        "a foreign model that bypasses the quarantine before-validator. Offending "
        "read paths:\n"
        + "\n".join(
            f"  {o.path}:{o.lineno} -> {o.model}.model_validate(...) [{o.snippet}]"
            for o in offenders
        )
    )


def test_lint_flags_a_synthetic_bypassing_read_path() -> None:
    """The lint is not vacuous: a comprehension parsing a tools-array row value
    into a NON-ToolSpec model is detected, while the sanctioned ToolSpec form is
    not.

    Exercises the AST detector directly on synthetic source so the guard's teeth
    are proven without committing a real bypass to the tree.
    """

    import ast

    from aios.retirements import read_paths

    bypass_src = "def read(row):\n    return [BareTool.model_validate(d) for d in row['tools']]\n"
    sanctioned_src = (
        "def read(row):\n    return [ToolSpec.model_validate(d) for d in row['tools']]\n"
    )
    loader_src = "def read(row):\n    return load_tool_specs(row['tools'])\n"

    def _foreign_models(src: str) -> list[str]:
        tree = ast.parse(src)
        out: list[str] = []
        for node in ast.walk(tree):
            iterables: list[ast.AST] = []
            calls: list[ast.Call] = []
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                iterables = [g.iter for g in node.generators]
                if isinstance(node.elt, ast.Call):
                    calls = [node.elt]
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                iterables = [node.iter]
                calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
            else:
                continue
            if not any(read_paths._is_tools_iterable(it) for it in iterables):
                continue
            for call in calls:
                model = read_paths._foreign_validate_model(call)
                if model is not None:
                    out.append(model)
        return out

    assert _foreign_models(bypass_src) == ["BareTool"]
    assert _foreign_models(sanctioned_src) == []  # ToolSpec is the sanctioned form
    assert _foreign_models(loader_src) == []  # load_tool_specs is the loader form


def test_lint_does_not_ban_per_site_toolspec_model_validate() -> None:
    """The rule is NOT "ban model_validate outside a loader".

    The connection tool-provider read at ``harness/step_context.py`` does
    ``ToolSpec.model_validate(d) for d in connection_tool_dicts`` — a deliberate
    per-site validator-resident tolerance point. The lint must leave it (and
    every other ``ToolSpec.model_validate`` site) alone; only FOREIGN models are
    flagged.
    """

    # No offender names ToolSpec — the sanctioned per-site form is untouched.
    for offender in iter_foreign_toolspec_parses():
        assert offender.model != "ToolSpec"


# ---------------------------------------------------------------------------
# Part B — reflective surface-coverage.
# ---------------------------------------------------------------------------


def test_every_consumed_toolspec_column_is_a_registered_surface() -> None:
    """ACCEPTANCE: a JSONB column consumed as ``ToolSpec`` but absent from every
    descriptor's surface list FAILS CI.

    Reflective: the consumed set is derived from the SQL→ToolSpec call-graph
    linkage, NOT a hand-list. The current seven-surface registry covers every
    consumed column, so this passes today; a new ToolSpec-consuming column added
    without a registry surface would fail here.
    """

    consumed = {(c.table, c.column) for c in iter_toolspec_consumed_columns()}
    registered = surface_columns()
    missing = consumed - registered
    assert missing == set(), (
        "These JSONB columns are consumed as ToolSpec but missing from every "
        f"descriptor's surface list (close them in the registry): {sorted(missing)}"
    )


def test_coverage_includes_the_seventh_surface_connectors_tools_schema() -> None:
    """The check keys on consumed-as-ToolSpec, so it sees ``connectors.tools_schema``.

    ``cat.tools_schema AS tools`` is selected in the connection tool-provider and
    fed to ``ToolSpec.model_validate`` in the per-step prelude — even though the
    column is named ``tools_schema``, not ``tools``. The introspection must
    resolve the alias back to its true ``(connectors, tools_schema)``.
    """

    consumed = {(c.table, c.column) for c in iter_toolspec_consumed_columns()}
    assert SEVENTH_SURFACE in consumed, (
        "connectors.tools_schema must be detected as consumed-as-ToolSpec "
        f"(found: {sorted(consumed)})"
    )


def test_coverage_test_would_have_flagged_connectors_tools_schema() -> None:
    """ACCEPTANCE: validated against a registry that — like the ad-hoc #1419
    retirement — enumerates only the first SIX surfaces, the coverage check
    FLAGS ``connectors.tools_schema`` as the missing seventh.

    This is the regression the whole epic exists to foreclose: it proves the
    check has teeth against the exact historical omission.
    """

    six_surfaces = tuple(s for s in reg.TOOL_SURFACES if s.table != "connectors")
    assert len(six_surfaces) == 6
    six_registry = tuple(dataclasses.replace(r, surfaces=six_surfaces) for r in reg.REGISTRY)

    registered_six = surface_columns(six_registry)
    consumed = {(c.table, c.column) for c in iter_toolspec_consumed_columns()}
    missing = consumed - registered_six

    assert SEVENTH_SURFACE in missing, (
        "Against the six-surface (pre-#1577) registry the coverage check must "
        f"flag connectors.tools_schema as missing (flagged: {sorted(missing)})"
    )


def test_surface_columns_is_the_descriptor_union_not_a_hand_list() -> None:
    """The Part B oracle is derived from the descriptors, not a constant."""

    expected = {(s.table, s.jsonb_col) for r in reg.REGISTRY for s in r.surfaces}
    assert SEVENTH_SURFACE in expected
    assert surface_columns() == expected


# ---------------------------------------------------------------------------
# SQL alias-resolution unit coverage (the load-bearing seventh-surface path).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql,expected",
    [
        # Bare column off a single FROM table.
        (
            "SELECT tools, mcp_servers FROM sessions WHERE id = $1",
            {("sessions", "tools")},
        ),
        # Aliased seventh-surface form: resolve cat -> connectors.
        (
            "SELECT cat.tools_schema AS tools FROM connectors cat WHERE cat.connector = $1",
            {("connectors", "tools_schema")},
        ),
        # Aliased form across a JOIN: the alias of the selected column wins.
        (
            "SELECT cat.tools_schema AS tools, s.focal_channel "
            "FROM connections c JOIN connectors cat ON cat.connector = c.connector "
            "JOIN sessions s ON s.id = $3",
            {("connectors", "tools_schema")},
        ),
        # No tools column → nothing.
        ("SELECT depth FROM wf_runs WHERE id = $1", set()),
        # JSON-path projection must not match the bare ``tools`` word.
        ("SELECT e.data->>'tool_call_id' AS tool_call_id FROM events e", set()),
    ],
)
def test_columns_from_sql_resolves_table_and_alias(
    sql: str, expected: set[tuple[str, str]]
) -> None:
    got = {(c.table, c.column) for c in _columns_from_sql(sql)}
    assert got == expected


def test_consumed_column_dataclass_is_descriptive() -> None:
    cols = _columns_from_sql(
        "SELECT cat.tools_schema AS tools FROM connectors cat WHERE cat.connector = $1"
    )
    assert cols == [
        ConsumedColumn(
            table="connectors",
            column="tools_schema",
            via_alias="tools",
            sql="SELECT cat.tools_schema AS tools FROM connectors cat WHERE cat.connector = $1",
        )
    ]


# ---------------------------------------------------------------------------
# Guard: the introspection oracle must agree with the registry test's expected
# seven-surface set, so the two CI checks can never silently diverge.
# ---------------------------------------------------------------------------


def test_introspection_oracle_matches_seven_surface_registry() -> None:
    assert surface_columns() == {
        ("agents", "tools"),
        ("agent_versions", "tools"),
        ("workflows", "tools"),
        ("workflow_versions", "tools"),
        ("wf_runs", "tools"),
        ("sessions", "tools"),
        SEVENTH_SURFACE,
    }
