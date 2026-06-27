"""Read-path introspection for the CI surface-coverage checks (#1577).

This module is the static-analysis core consumed by the two CI read-path
coverage checks (epic #1572) that together close the
"an unregistered raw-dict surface is invisible to both the metric and the guard"
blind spot:

* **Part A — AST-lint** (:func:`iter_foreign_toolspec_parses`): asserts that a
  persisted ``tools`` / ``tools_schema`` JSONB array element is **only ever
  parsed into a** :class:`~aios.models.agents.ToolSpec` — never validated into
  some *other* Pydantic model that would bypass the quarantine before-validator
  (and thus the registry-driven read-tolerance + the boot scan's coverage). The
  rule is aimed at the RIGHT thing: it is **NOT** "ban ``model_validate`` outside
  a loader" (that would break the working per-site tolerance, which deliberately
  lives at every ``ToolSpec.model_validate`` site). It fails the build only when
  a read path materialises a persisted tools array into a model OTHER than
  ``ToolSpec``.

* **Part B — Reflective surface-coverage** (:func:`iter_toolspec_consumed_columns`):
  enumerates every JSONB column **consumed as a ``ToolSpec``** — by the SQL that
  feeds a ``ToolSpec`` read site, NOT a hand-list — so a column wired into a
  ``ToolSpec`` consumer but absent from every descriptor's surface list is
  caught. It keys on "consumed-as-ToolSpec", which is exactly why it flags
  ``connectors.tools_schema`` (selected ``cat.tools_schema AS tools`` and fed to
  ``ToolSpec.model_validate`` through the connection tool-provider) as the
  seventh surface.

Both checks are pure static analysis (stdlib :mod:`ast` + a small SQL
``SELECT``-column scan) over the in-tree source — no DB, no import side effects —
so they run in the fast unit/lint tier alongside ``test_append_event_structure``
and friends.

Residual (carried forward, ops-agent-visible): a future surface that stores the
token WITHOUT going through ``ToolSpec`` (a raw-dict consumer added in violation
of the lint, or a non-Postgres store) stays invisible until the lint/test catches
the code or the ops-agent notices. Surface-completeness is the irreducible
residual.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from aios.retirements import Retirement
from aios.retirements.registry import REGISTRY

# ---------------------------------------------------------------------------
# Where to look.
# ---------------------------------------------------------------------------

#: Root of the importable ``aios`` package (``.../src/aios``).
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

#: The canonical loader + per-``ToolSpec`` consumer names. A persisted tools
#: array element is "parsed into a ToolSpec" when it flows through one of these.
#: ``load_tool_specs`` is the list-level loader (drops retired-builtin entries,
#: then ``ToolSpec.model_validate`` each remaining); ``ToolSpec.model_validate`` /
#: ``ToolSpec.model_validate_json`` are the per-site form the tolerance validator
#: rides on.
TOOLSPEC_MODEL = "ToolSpec"
TOOLSPEC_LOADER = "load_tool_specs"
_TOOLSPEC_VALIDATE_METHODS = frozenset({"model_validate", "model_validate_json"})


# ---------------------------------------------------------------------------
# Surface list (Part B's oracle) — derived from the descriptors, not hand-listed.
# ---------------------------------------------------------------------------


def surface_columns(
    registry: tuple[Retirement, ...] = REGISTRY,
) -> frozenset[tuple[str, str]]:
    """``{(table, jsonb_col)}`` declared on *every* descriptor's surface list.

    This is the registry-side oracle the surface-coverage test validates the
    code-derived set against. It is the union over all descriptors, so a column
    declared on any descriptor counts as a registered surface.
    """

    out: set[tuple[str, str]] = set()
    for retirement in registry:
        for surface in retirement.surfaces:
            out.add((surface.table, surface.jsonb_col))
    return frozenset(out)


# ---------------------------------------------------------------------------
# Source iteration.
# ---------------------------------------------------------------------------


def _python_sources(root: Path = PACKAGE_ROOT) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, source)`` for every ``.py`` under ``root``."""

    for path in sorted(root.rglob("*.py")):
        yield path, path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# SQL SELECT-column scan: which (table, column) is "consumed as a ToolSpec".
# ---------------------------------------------------------------------------

# A persisted tools-array column is selected either bare (``SELECT tools, ...``)
# or aliased to the conventional ``tools`` name (``SELECT cat.tools_schema AS
# tools``). The alias form is the load-bearing one: it is how the seventh
# surface (``connectors.tools_schema``) reaches a ToolSpec consumer under the
# ``tools`` row key. We resolve the alias back to its true ``(table, column)``.

#: ``<table-or-alias>.<column> AS tools`` — e.g. ``cat.tools_schema AS tools``.
_ALIASED_TOOLS_RE = re.compile(
    r"(?P<qual>[A-Za-z_][A-Za-z0-9_]*)\.(?P<col>[A-Za-z_][A-Za-z0-9_]*)\s+AS\s+tools\b",
    re.IGNORECASE,
)

#: ``FROM <table> [AS] <alias>`` / ``JOIN <table> [AS] <alias>`` — used to map a
#: query alias (``cat``) back to its base table (``connectors``).
_FROM_JOIN_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)

#: ``SELECT tools`` / ``SELECT tools,`` — a bare ``tools`` column off the single
#: ``FROM`` table. Excludes ``e.data->>'...'`` JSON-path projections and aliases.
_BARE_TOOLS_RE = re.compile(r"\bSELECT\b(?P<cols>.*?)\bFROM\b", re.IGNORECASE | re.DOTALL)

#: SQL keywords that must NOT be treated as a query alias (e.g. ``FROM connectors
#: WHERE`` — ``WHERE`` is not an alias of ``connectors``).
_NOT_AN_ALIAS = frozenset(
    {"where", "on", "join", "inner", "left", "right", "full", "cross", "group", "order", "having"}
)


@dataclass(frozen=True)
class ConsumedColumn:
    """A JSONB ``(table, column)`` that a SQL ``SELECT`` exposes as ToolSpec rows.

    ``via_alias`` records the ``... AS tools`` alias when present (the seventh
    surface is reached this way); ``sql`` is the originating query (trimmed) for
    diagnostics.
    """

    table: str
    column: str
    via_alias: str | None
    sql: str


def _alias_table_map(sql: str) -> dict[str, str]:
    """``alias -> table`` (and ``table -> table``) for every FROM/JOIN in ``sql``."""

    out: dict[str, str] = {}
    for m in _FROM_JOIN_RE.finditer(sql):
        table = m.group("table")
        alias = m.group("alias")
        out[table] = table
        if alias and alias.lower() not in _NOT_AN_ALIAS:
            out[alias] = table
    return out


def _columns_from_sql(sql: str) -> list[ConsumedColumn]:
    """Resolve every tools-array column a single SQL ``SELECT`` exposes as ``tools``.

    Handles both the aliased form (``cat.tools_schema AS tools`` → resolve ``cat``
    to its base table) and the bare form (``SELECT tools FROM sessions``). A
    ``SELECT *`` does not name ``tools`` and yields nothing here — those read
    sites are covered by the AST linkage in :func:`iter_toolspec_consumed_columns`
    falling back to the model layer, but in practice every ToolSpec-feeding query
    in-tree names the column. JSON-path projections (``data->>'tool_call_id'``)
    never match the ``\\btools\\b`` column word.
    """

    if "tools" not in sql.lower():
        return []
    alias_map = _alias_table_map(sql)
    out: list[ConsumedColumn] = []

    # Aliased: ``<qual>.<col> AS tools``.
    for m in _ALIASED_TOOLS_RE.finditer(sql):
        qual = m.group("qual")
        col = m.group("col")
        table = alias_map.get(qual, qual)
        out.append(ConsumedColumn(table=table, column=col, via_alias="tools", sql=sql.strip()))

    # Bare: a column literally named ``tools`` in the SELECT list, off the single
    # base table. Only trust this when the query has exactly one base table (no
    # ambiguity about which relation ``tools`` belongs to). Strip the aliased
    # ``<qual>.<col> AS tools`` segments first so their ``tools`` *alias* is not
    # mistaken for a bare ``tools`` column (already handled above).
    select_cols_match = _BARE_TOOLS_RE.search(sql)
    if select_cols_match:
        cols_blob = _ALIASED_TOOLS_RE.sub(" ", select_cols_match.group("cols"))
        col_names = {c.strip() for c in re.split(r"[,\s]+", cols_blob) if c.strip()}
        if "tools" in {c.lower() for c in col_names}:
            base_tables = {t for t in alias_map.values()}
            if len(base_tables) == 1:
                (table,) = tuple(base_tables)
                out.append(
                    ConsumedColumn(table=table, column="tools", via_alias=None, sql=sql.strip())
                )

    return out


# ---------------------------------------------------------------------------
# Part B — reflective surface-coverage: columns consumed as ToolSpec.
# ---------------------------------------------------------------------------


def _call_callee_name(call: ast.Call) -> str | None:
    """The simple callee name of a ``Call`` (``f()``→``f``, ``a.f()``→``f``)."""

    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


@dataclass(frozen=True)
class _FuncFacts:
    """Per-function static facts feeding the call-graph reachability scan."""

    name: str
    calls: frozenset[str]
    consumes_toolspec: bool
    columns: tuple[ConsumedColumn, ...]


def _is_toolspec_consume_call(call: ast.Call) -> bool:
    """True if ``call`` materialises a persisted tools array into ``ToolSpec``."""

    func = call.func
    if isinstance(func, ast.Name) and func.id == TOOLSPEC_LOADER:
        return True
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _TOOLSPEC_VALIDATE_METHODS
        and isinstance(func.value, ast.Name)
        and func.value.id == TOOLSPEC_MODEL
    )


def _facts_for_scope(name: str, body_nodes: list[ast.AST]) -> _FuncFacts:
    """Build :class:`_FuncFacts` from the nodes directly owned by a scope.

    ``body_nodes`` are the nodes belonging to ``name`` *excluding* the bodies of
    nested functions (those are scanned under their own name), so a column or a
    consume in an inner ``def`` is not double-attributed to its enclosing scope.
    """

    calls: set[str] = set()
    consumes = False
    columns: list[ConsumedColumn] = []
    for node in body_nodes:
        if isinstance(node, ast.Call):
            callee = _call_callee_name(node)
            if callee is not None:
                calls.add(callee)
            if _is_toolspec_consume_call(node):
                consumes = True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            columns.extend(_columns_from_sql(node.value))
    return _FuncFacts(
        name=name,
        calls=frozenset(calls),
        consumes_toolspec=consumes,
        columns=tuple(columns),
    )


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """All descendant nodes of ``scope`` that are NOT inside a nested function.

    Nested ``def``/``async def`` headers are included (so their call edge is
    attributed to the enclosing scope) but their *bodies* are excluded.
    """

    out: list[ast.AST] = []
    for child in ast.iter_child_nodes(scope):
        out.append(child)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue  # body scanned under the nested function's own name
        out.extend(_own_nodes(child))
    return out


def _scan_functions(root: Path = PACKAGE_ROOT) -> list[_FuncFacts]:
    """Collect per-function facts across the package.

    For every ``def`` / ``async def`` (and a synthetic per-module ``<module>``
    scope for top-level code) we record: the callee names it invokes, whether it
    materialises a persisted tools array into ``ToolSpec``, and the tools-array
    columns its inline SQL ``SELECT``\\ s expose. Nested functions are scanned as
    their own scope so a column never double-attributes to an enclosing function.
    """

    facts: list[_FuncFacts] = []
    for _path, source in _python_sources(root):
        tree = ast.parse(source)

        functions: list[ast.AST] = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for fn in functions:
            assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            facts.append(_facts_for_scope(fn.name, _own_nodes(fn)))

        # Module-level (top-level) code, excluding function bodies.
        facts.append(_facts_for_scope("<module>", _own_nodes(tree)))
    return facts


def _toolspec_reachable_funcs(facts: list[_FuncFacts]) -> set[str]:
    """Function names reachable (by callee name) from any ToolSpec consumer.

    A tools-array column is consumed-as-ToolSpec when the function that selects
    it is reachable, through the call graph, from a function that materialises a
    persisted tools array into ``ToolSpec``. This is what links the seventh
    surface — ``connectors.tools_schema`` is selected in
    ``list_connection_tools_for_session``, which the prelude reaches (via the
    ``ToolProvider`` seam) before ``ToolSpec.model_validate`` — while *excluding*
    the connector fan-out functions that select the same column but feed only
    bare-dict field reads (never reachable from a ToolSpec consumer).

    Edges are keyed on simple callee name (``a.f()`` and ``f()`` both → ``f``),
    which is coarse but conservative: it can only ever *widen* the consumed set,
    so the surface-coverage check stays sound (it never under-reports a column
    that should be registered).
    """

    callers_consume = {f.name for f in facts if f.consumes_toolspec}
    callees_by_caller: dict[str, set[str]] = {}
    for f in facts:
        callees_by_caller.setdefault(f.name, set()).update(f.calls)

    reachable: set[str] = set(callers_consume)
    frontier = list(callers_consume)
    while frontier:
        current = frontier.pop()
        for callee in callees_by_caller.get(current, ()):
            if callee not in reachable:
                reachable.add(callee)
                frontier.append(callee)
    return reachable


def iter_toolspec_consumed_columns(
    root: Path = PACKAGE_ROOT,
) -> Iterator[ConsumedColumn]:
    """Yield every JSONB ``(table, column)`` *consumed as a ``ToolSpec``* in-tree.

    Keyed on *consumption shape*, not column type or a hand-list: a tools-array
    column counts when the function that ``SELECT``\\ s it is reachable, through
    the call graph, from a function that materialises a persisted tools array
    into ``ToolSpec`` (``load_tool_specs`` / ``ToolSpec.model_validate[_json]``).

    This flags ``connectors.tools_schema`` as the seventh surface — it is
    selected ``cat.tools_schema AS tools`` in ``list_connection_tools_for_session``
    and fed through the connection tool-provider to ``ToolSpec.model_validate`` —
    while NOT flagging the connector tool-call fan-out reads of the same column,
    which consume bare dicts and are never reachable from a ToolSpec consumer.
    """

    facts = _scan_functions(root)
    reachable = _toolspec_reachable_funcs(facts)
    seen: set[tuple[str, str, str | None]] = set()
    for f in facts:
        if f.name not in reachable:
            continue
        for col in f.columns:
            key = (col.table, col.column, col.via_alias)
            if key in seen:
                continue
            seen.add(key)
            yield col


# ---------------------------------------------------------------------------
# Part A — AST-lint: persisted tools parsed into anything OTHER than ToolSpec.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForeignParse:
    """A read site validating a persisted tools array element into a non-ToolSpec.

    ``model`` is the offending Pydantic model name; ``path``/``lineno`` locate
    the call. This is the bypass Part A forecloses: a foreign model resident over
    a persisted tools array dodges the quarantine before-validator (and thus the
    registry-driven tolerance + boot-scan coverage).
    """

    path: Path
    lineno: int
    model: str
    snippet: str


# A loop/comprehension variable bound by iterating a persisted tools/tools_schema
# row value, e.g. ``for d in connection_tool_dicts`` where the dicts came from a
# ``SELECT ... AS tools`` row. We detect the *direct* shape the read paths use:
# ``Model.model_validate(<elt>) for <elt> in <iterable>`` and
# ``for <elt> in <iterable>: ... Model.model_validate(<elt>)`` where ``<iterable>``
# is a tools-array row value (``row["tools"]``, a name ending in ``tool_dicts`` /
# ``tools_list`` / ``tools_data``, or the loader-shaped ``raw`` argument).

_TOOLS_ITERABLE_NAME_RE = re.compile(r"tool_dicts$|tools_list$|tools_data$|^raw$|^tools$")


def _is_tools_row_subscript(node: ast.AST) -> bool:
    """True for ``row["tools"]`` / ``row["tools_schema"]`` style subscripts."""

    if not isinstance(node, ast.Subscript):
        return False
    key = node.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value in {"tools", "tools_schema"}
    return False


def _is_tools_iterable(node: ast.AST) -> bool:
    """True if ``node`` is a persisted tools-array iterable expression."""

    if _is_tools_row_subscript(node):
        return True
    if isinstance(node, ast.Name) and _TOOLS_ITERABLE_NAME_RE.search(node.id):
        return True
    # ``tools_list or []`` / ``row["tools"] or []`` — the common null-guard form.
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return any(_is_tools_iterable(v) for v in node.values)
    return False


def _foreign_validate_model(call: ast.Call) -> str | None:
    """Return the model name if ``call`` is ``<Model>.model_validate[_json](...)``
    with ``<Model>`` != ``ToolSpec``; else ``None``.
    """

    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr in _TOOLSPEC_VALIDATE_METHODS):
        return None
    if not isinstance(func.value, ast.Name):
        return None
    model = func.value.id
    if model == TOOLSPEC_MODEL:
        return None
    return model


def iter_foreign_toolspec_parses(root: Path = PACKAGE_ROOT) -> Iterator[ForeignParse]:
    """Yield read sites parsing a persisted tools array element into a non-ToolSpec.

    The lint rule (Part A): a persisted ``tools`` / ``tools_schema`` array element
    may be validated into a Pydantic model ONLY if that model is ``ToolSpec``. A
    comprehension or for-loop that iterates a tools-array row value and calls
    ``<Model>.model_validate(elt)`` with ``<Model>`` other than ``ToolSpec`` is a
    bypass of the quarantine before-validator and FAILS the build.

    This deliberately does NOT flag:

    * ``ToolSpec.model_validate(...)`` / ``load_tool_specs(...)`` — the sanctioned
      per-site tolerance path (banning *those* is the explicitly-rejected
      "ban model_validate outside a loader").
    * bare-dict field reads over a tools array (``t["name"]`` for connector
      tool-call fan-out gating) — those don't *parse into a model* at all; they
      read a field. The lint targets model materialisation, the only way a
      typed read path can dodge the validator.
    """

    for path, source in _python_sources(root):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            iterables: list[ast.AST] = []
            calls: list[ast.Call] = []

            if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                iterables = [gen.iter for gen in node.generators]
                if isinstance(node.elt, ast.Call):
                    calls = [node.elt]
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                iterables = [node.iter]
                calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
            else:
                continue

            if not any(_is_tools_iterable(it) for it in iterables):
                continue

            for call in calls:
                model = _foreign_validate_model(call)
                if model is None:
                    continue
                yield ForeignParse(
                    path=path,
                    lineno=getattr(call, "lineno", getattr(node, "lineno", 0)),
                    model=model,
                    snippet=ast.unparse(call),
                )
