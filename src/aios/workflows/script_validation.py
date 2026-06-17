"""Create-time validation of a workflow ``script`` (#1221 / #1286).

A workflow is authored out-of-band as an inert ``script: str`` whose only contract
is a docstring, and whose declared tool/server surface is authored *separately* from
the script body. Historically three failure classes only surfaced as a *failed run*
read back from the journal rather than as an authoring-time error:

1. **Syntax / structure errors** — a typo, or a missing top-level ``async def
   main(input)`` — not caught until the host ``exec``'d the script in a run.
2. **Silent surface drift** — the declared ``tools`` / ``http_servers`` /
   ``mcp_servers`` must be the **union** of the script's own ``tool("…")`` calls and
   every named ``agent(agent_id="…")`` child's tools; an under-declaration is
   silently stripped by the #794 clamp and detonates as a runtime route-mismatch.

This module lifts the host's own ``compile(...)`` to create time and adds an AST
check, closing classes 1 and 2 (value-domain traps — class 3 — are out of scope,
tracked by #934).

**Design (SETTLED, #1286): validate-declared, literal-only extraction.** The author
writes the declared surface; the validator checks the declared surface is a
**superset** of the AST-extracted *required* surface. Only string-literal
``tool("…")`` names and string-literal ``agent(agent_id="…")`` references participate
in the required-surface union. A ``tool(name_var, …)`` or an ``agent_id`` that is not
a string literal (computed / variable / from ``input``) is **un-AST-able** and is
**excluded** from the check — it never causes a rejection.

The single public entry point is :func:`validate_workflow_script`. It is pure and
synchronous over the AST; the *named-agent surface* (the ``agent_id`` participation)
is resolved by an injected ``resolve_agent_tools`` callback so the AST core stays
DB-free and unit-testable, while the service path wires in the real agent lookup.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field

from aios.errors import ValidationError
from aios.models.agents import ToolSpec

# An ``agent_id``-surface resolver: given a literal agent id, return the set of tool
# names (``ToolSpec`` identity keys, see :func:`_declared_tool_names`) that the named
# child agent would bring to the run. Returning ``None`` means "agent not found /
# unresolvable" — its surface is then excluded from the union (un-resolvable, like an
# un-AST-able name), never a false rejection. The service supplies a DB-backed
# implementation; unit tests supply a stub.
ResolveAgentTools = Callable[[str], "frozenset[str] | None"]


@dataclass(frozen=True)
class _RequiredSurface:
    """The AST-extracted *required* surface: literal tool names + literal agent ids."""

    tool_names: set[str] = field(default_factory=set)
    agent_ids: set[str] = field(default_factory=set)


def _string_literal(node: ast.expr | None) -> str | None:
    """Return the ``str`` value if ``node`` is a string-literal constant, else ``None``
    (the un-AST-able case: a Name, an attribute, a subscript of ``input``, a join, …)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_func_name(node: ast.Call) -> str | None:
    """The bare callable name for a ``Call`` node (``tool(...)`` → ``"tool"``).

    Only a direct ``Name`` callee counts — ``mod.tool(...)`` (an attribute call) is not
    the injected builtin and is ignored. The author namespace injects ``tool`` / ``agent``
    as bare names, so this is the exact match for the capability API.
    """
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _extract_required_surface(tree: ast.Module) -> _RequiredSurface:
    """Walk the whole module and collect literal ``tool("…")`` names and literal
    ``agent(agent_id="…")`` ids. Non-literal arguments are silently excluded."""
    required = _RequiredSurface()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _call_func_name(node)
        if fname == "tool":
            # tool(name, input) — name is the first positional argument.
            if node.args:
                name = _string_literal(node.args[0])
                if name is not None:
                    required.tool_names.add(name)
            else:
                # Defensively also accept a keyword ``tool(name="…", …)``.
                for kw in node.keywords:
                    if kw.arg == "name":
                        name = _string_literal(kw.value)
                        if name is not None:
                            required.tool_names.add(name)
        elif fname == "agent":
            # agent(input, *, agent_id="…", …) — agent_id is keyword-only.
            for kw in node.keywords:
                if kw.arg == "agent_id":
                    aid = _string_literal(kw.value)
                    if aid is not None:
                        required.agent_ids.add(aid)
    return required


def extract_literal_agent_ids(script: str) -> set[str]:
    """Return the set of string-literal ``agent(agent_id="…")`` ids in ``script``.

    A convenience for callers (the service layer) that need to *pre-resolve* each named
    agent's surface asynchronously before handing the pure validator a sync resolver. An
    unparsable script yields the empty set (the compile step reports the syntax error).
    """
    try:
        tree = ast.parse(script, "<workflow>")
    except SyntaxError:
        return set()
    return _extract_required_surface(tree).agent_ids


def _declared_tool_names(tools: list[ToolSpec]) -> set[str]:
    """The names a declared tool surface satisfies for a script ``tool("X")`` call.

    A run resolves ``tool("X")`` against ``{t.type for t in run.tools}`` (see
    :func:`aios.workflows.run_tools.gate_run_tool`), so ``type`` is the load-bearing
    key (``"bash"``, ``"http_request"``, …). ``name`` is also accepted so a custom
    tool referenced by its author-facing name is not a false positive.
    """
    names: set[str] = set()
    for t in tools:
        names.add(t.type)
        if t.name:
            names.add(t.name)
        if t.mcp_server_name:
            names.add(t.mcp_server_name)
    return names


def _find_main(tree: ast.Module) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    """Return the top-level (module-level) function named ``main``, async or not, or
    ``None`` if there is no top-level ``main``."""
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == "main":
            return node
    return None


def _check_main_signature(fn: ast.AsyncFunctionDef | ast.FunctionDef) -> None:
    """Raise :class:`ValidationError` unless ``fn`` can be called as ``async def
    main(input)`` — i.e. it is ``async`` and accepts exactly one positional argument.

    The host invokes ``main(input_value)`` *positionally* (see
    ``aios.workflows.wf_script_host._build_coroutine``), so the parameter's NAME is not
    load-bearing — ``async def main(i)`` is admissible. What matters is arity: it must
    accept exactly the single positional ``input`` and nothing more. A zero-arg
    ``main()``, a two-arg ``main(a, b)``, a keyword-only ``main(*, input)``, or a
    ``*args``/``**kwargs`` shape (which would receive ``input`` in the wrong slot, or a
    required extra arg the host never supplies) is rejected.
    """
    if not isinstance(fn, ast.AsyncFunctionDef):
        raise ValidationError(
            "workflow script `main` must be `async def main(input)`: the top-level "
            "`main` is a plain `def`, not `async def`."
        )
    a = fn.args
    positional = a.posonlyargs + a.args
    # The host calls ``main(input_value)`` positionally, so: exactly one positional
    # parameter, no *args/**kwargs, and no keyword-only params (the single positional
    # ``input`` could not be supplied by name to a keyword-only-only signature).
    has_extra = bool(a.vararg or a.kwarg or a.kwonlyargs)
    if len(positional) != 1 or has_extra:
        raise ValidationError(
            "workflow script `main` must be `async def main(input)` accepting exactly "
            "the single positional `input` parameter (the run's input is passed "
            "positionally)."
        )


def validate_workflow_script(
    script: str,
    *,
    tools: list[ToolSpec] | None = None,
    resolve_agent_tools: ResolveAgentTools | None = None,
) -> None:
    """Create/update-time validation of a workflow ``script``. Raises
    :class:`ValidationError` (a client-class 4xx) with a precise, actionable message on
    the first failure; returns ``None`` when the script is admissible.

    Checks, in order:

    1. **Compile** — ``compile(script, "<workflow>", "exec", dont_inherit=True)`` (the
       exact call the host makes at run time, lifted here). A ``SyntaxError`` surfaces
       as a compile/syntax validation error naming the line/offset where available.
    2. **`main` present** — a top-level function named ``main`` must exist.
    3. **`main` signature** — it must be ``async def main(input)`` (async, accepting
       exactly one positional parameter; the host calls it positionally, so the
       parameter name is not load-bearing).
    4. **Tool surface covered** — every string-literal ``tool("X")`` name must be in the
       declared tool surface (by ``type``/``name``).
    5. **Named-agent surface covered** — every string-literal ``agent(agent_id="A")``
       child's tool surface (resolved via ``resolve_agent_tools``) must be covered by
       the declared tool surface. When no resolver is supplied, or the agent does not
       resolve, the agent's surface is excluded from the union (never a false rejection).

    Un-AST-able names (a computed/variable ``tool`` name or ``agent_id``) are excluded
    from steps 4-5 by construction (they are not literals, so never extracted).
    """
    # 1. Compile (the host's own call, lifted to author time).
    try:
        compile(script, "<workflow>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        where = ""
        if exc.lineno is not None:
            where = f" (line {exc.lineno}"
            if exc.offset is not None:
                where += f", offset {exc.offset}"
            where += ")"
        detail = (exc.msg or "syntax error").strip()
        raise ValidationError(f"workflow script failed to compile: {detail}{where}") from exc

    # Re-parse to an AST for the structural checks. ``compile`` already proved it parses,
    # so ``ast.parse`` cannot raise here.
    tree = ast.parse(script, "<workflow>")

    # 2/3. main present + correctly signatured.
    fn = _find_main(tree)
    if fn is None:
        raise ValidationError(
            "workflow script must define a top-level `async def main(input)` "
            "(no top-level `main` was found)."
        )
    _check_main_signature(fn)

    # 4/5. Surface coverage.
    required = _extract_required_surface(tree)
    declared = _declared_tool_names(tools or [])

    missing_tools = sorted(required.tool_names - declared)
    if missing_tools:
        names = ", ".join(repr(t) for t in missing_tools)
        raise ValidationError(
            "workflow script declares a tool surface that under-covers its "
            f"`tool(...)` calls: missing tool(s) {names}. Add them to the workflow's "
            "declared `tools` (or pass a non-literal name to opt out of the check).",
            detail={"missing_tools": missing_tools},
        )

    if resolve_agent_tools is not None:
        missing_from_agents: set[str] = set()
        for agent_id in sorted(required.agent_ids):
            agent_tools = resolve_agent_tools(agent_id)
            if agent_tools is None:
                # Unresolvable named agent: excluded from the union (like an un-AST-able
                # name), never a false rejection.
                continue
            missing_from_agents |= agent_tools - declared
        if missing_from_agents:
            names = ", ".join(repr(t) for t in sorted(missing_from_agents))
            raise ValidationError(
                "workflow script references a named agent whose tool surface is not "
                f"covered by the declared `tools`: missing element(s) {names}. The "
                "#794 clamp would silently strip these from the child. Add them to the "
                "workflow's declared `tools`.",
                detail={"missing_agent_surface": sorted(missing_from_agents)},
            )
