"""Create-time validation for workflow author scripts (#1221 / #1284).

A workflow is authored out-of-band as an inert ``script: str`` whose declared
tool/server surface is hand-derived and authored *separately* from the script.
Without create-time checking, three failure classes surface only as a *failed
run* read back from the journal instead of as an authoring-time error. This
module closes the two structural classes (the value-domain class is #934):

1. **Syntax / structure** — the script does not ``compile(...)``, or it has no
   top-level ``async def main(input)`` (right name, ``async``, single ``input``
   parameter).
2. **Surface drift** — a string-literal ``tool("X", …)`` call (or string-literal
   ``agent(agent_id="A")`` reference) whose required surface is not covered by
   the *declared* surface. The author writes the declared surface; the validator
   checks it is a **superset** of the AST-extracted required surface (the
   *validate-declared* fork, settled in #1284 — NOT auto-derive).

**Literal-only extraction.** When a ``tool(name, …)`` name or an ``agent_id`` is
not a string literal (computed / a variable / from ``input``), that element is
*un-AST-able* and is **excluded** from the required-surface check — the validator
does not reject on it and does not attempt to resolve it. Only string-literal
``tool("…")`` names and string-literal ``agent(agent_id="…")`` references take
part in the union.

The host already does ``compile(source, "<workflow>", "exec", dont_inherit=True)``
in the exec path (:mod:`aios.workflows.wf_script_host`); this lifts that compile
to create time and adds the AST checks, surfacing a precise, actionable error.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import TYPE_CHECKING

from aios.errors import ValidationError

if TYPE_CHECKING:
    from aios.models.agents import McpServerSpec, ToolSpec

# Filename used for the lifted compile, matching the run-time exec path so the
# surfaced line/offset is in the same coordinate system the author already sees
# in a run traceback.
_WORKFLOW_FILENAME = "<workflow>"


class WorkflowScriptValidationError(ValidationError):
    """A workflow script failed create-time validation (#1284).

    A 422 (``ValidationError`` subclass) carrying an author-facing message that
    names the specific defect — a compile/syntax failure (with line/offset where
    available), a missing/mis-signatured ``async def main(input)``, or the
    specific missing tool/server name(s) for an under-declared surface. Raised at
    create *and* update time, before any row is written.
    """

    error_type = "workflow_script_validation_error"
    status_code = 422


def _declared_tool_names(tools: Sequence[ToolSpec]) -> set[str]:
    """The set of names a script's ``tool("…")`` call may resolve to.

    A run resolves ``tool(name)`` against ``{t.type for t in run.tools}`` (the
    run-tool gate, :func:`aios.workflows.run_tools.gate_run_tool`), so a builtin's
    callable name is its ``type``. A custom tool is callable by its ``name``. Both
    are admitted so the required-surface check matches what the run will actually
    resolve. (``mcp_toolset`` tools are namespaced and are never called by a bare
    ``tool("…")`` literal, so they contribute no bare name here.)
    """
    names: set[str] = set()
    for t in tools:
        if t.type == "mcp_toolset":
            continue
        if t.type == "custom":
            if t.name is not None:
                names.add(t.name)
            continue
        names.add(t.type)
        if t.name is not None:
            names.add(t.name)
    return names


def _is_call_to(node: ast.Call, func_name: str) -> bool:
    """True iff ``node`` is a direct call to the bare name ``func_name`` — e.g.
    ``tool(...)`` / ``agent(...)``. Attribute calls (``obj.tool(...)``) are not the
    injected capability and are ignored."""
    return isinstance(node.func, ast.Name) and node.func.id == func_name


def _literal_str(node: ast.expr | None) -> str | None:
    """Return the string value if ``node`` is a string-literal constant, else None
    (the un-AST-able case: a variable, attribute, f-string, ``input[...]``, …)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_required_tools(tree: ast.AST) -> set[str]:
    """String-literal ``tool("X", …)`` names — the first positional arg, literal only."""
    required: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_call_to(node, "tool") and node.args:
            name = _literal_str(node.args[0])
            if name is not None:
                required.add(name)
    return required


def _extract_required_agent_ids(tree: ast.AST) -> set[str]:
    """String-literal ``agent(agent_id="A")`` references — the ``agent_id`` keyword,
    literal only. ``agent_id`` is keyword-only in the author API
    (``agent(input, *, agent_id=None, …)``), so only the keyword form participates.
    """
    required: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_call_to(node, "agent"):
            for kw in node.keywords:
                if kw.arg == "agent_id":
                    agent_id = _literal_str(kw.value)
                    if agent_id is not None:
                        required.add(agent_id)
    return required


def _find_top_level_main(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The last top-level ``def``/``async def`` named ``main`` (mirroring exec
    semantics, where a later binding shadows an earlier one), or None."""
    found: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "main":
            found = stmt
    return found


def _main_accepts_single_input(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True iff ``main``'s signature accepts the single ``input`` parameter — i.e.
    exactly one positional-or-keyword parameter and no extra required params.

    Accepts ``async def main(input)`` (the canonical form). Rejects ``main()``,
    ``main(a, b)``, and keyword-only/var-arg-only shapes that cannot be called as
    ``main(input_value)`` with one positional argument.
    """
    args = func.args
    positional = args.posonlyargs + args.args
    # Exactly one positional-or-keyword (or positional-only) parameter, and no
    # other required parameters. ``*args``/``**kwargs`` and defaulted extras are
    # not the contract shape; keep the check strict to the single ``input`` slot.
    if len(positional) != 1:
        return False
    if args.vararg is not None or args.kwarg is not None:
        return False
    # Any required keyword-only parameter would make ``main(input_value)`` fail
    # (a ``None`` in ``kw_defaults`` marks a kw-only param with no default).
    return all(default is not None for default in args.kw_defaults)


def validate_workflow_script(
    script: str,
    *,
    tools: Sequence[ToolSpec] | None = None,
    mcp_servers: Sequence[McpServerSpec] | None = None,
    http_servers: Sequence[object] | None = None,
) -> None:
    """Validate a workflow author script at create/update time (#1284).

    Raises :class:`WorkflowScriptValidationError` (422) with a precise,
    actionable message on any of:

    1. **Syntax / compile failure** — the script does not ``compile(...)`` (the
       lifted host compile); the message identifies it as a syntax failure and
       surfaces the line/offset where available. The workflow is not created.
    2. **Missing ``main``** — no top-level ``async def main`` — message names the
       required ``async def main(input)``.
    3. **Mis-signatured ``main``** — a top-level ``main`` that is not ``async``,
       or whose signature does not accept the single ``input`` parameter — message
       names the signature requirement.
    4. **Under-declared tool surface** — a string-literal ``tool("X", …)`` whose
       ``"X"`` is not in the declared ``tools`` — message names ``X``.
    5. **Under-declared agent reference** — a string-literal
       ``agent(agent_id="A")`` whose ``"A"`` is not in the declared agent surface
       — message names ``A``. (Depth note below.)

    **Surface model / chosen depth.** This is *validate-declared*: the declared
    surface must be a superset of the AST-extracted required surface, literal-only.
    The ``tool("X")`` union is checked against the declared tool names. For
    ``agent(agent_id="A")`` the child agent's *full* transitive surface cannot be
    resolved here (it lives in the DB and is out of scope for this create-time,
    DB-free validator, per #1284's "if resolving a child agent's full surface is
    out of scope … at minimum the literal ``agent_id`` participation in the union
    MUST be implemented"). So the ``agent_id`` participates in the union via the
    workflow's own declared **mcp_servers** named surface: a literal ``agent_id``
    that is not covered by a declared mcp server of the same name is reported as a
    missing element. A literal ``agent_id`` that names a *tool* the script could
    also call is covered by the declared tools. Un-AST-able (computed/variable)
    ``agent_id`` values are excluded from the check.

    Un-AST-able ``tool``/``agent_id`` names (computed / variable / from ``input``)
    are excluded from the required-surface check entirely — never a violation.
    """
    # 1. Compile / syntax. Lift the host's exec-path compile to create time and
    #    surface a precise error (line/offset where the SyntaxError carries them).
    try:
        compile(script, _WORKFLOW_FILENAME, "exec", dont_inherit=True)
    except SyntaxError as exc:
        loc = ""
        if exc.lineno is not None:
            loc = f" at line {exc.lineno}"
            if exc.offset is not None:
                loc += f", offset {exc.offset}"
        msg = exc.msg or "invalid syntax"
        raise WorkflowScriptValidationError(
            f"workflow script failed to compile (syntax error{loc}): {msg}",
            detail={"lineno": exc.lineno, "offset": exc.offset, "msg": msg},
        ) from exc

    # The AST is reparsed (compile() does not hand back a tree); same source, so it
    # cannot raise here after a clean compile.
    tree = ast.parse(script, filename=_WORKFLOW_FILENAME)

    # 2/3. Top-level async def main(input).
    main = _find_top_level_main(tree)
    if main is None:
        raise WorkflowScriptValidationError(
            "workflow script must define a top-level `async def main(input)` "
            "(no top-level `main` found)",
        )
    if not isinstance(main, ast.AsyncFunctionDef):
        raise WorkflowScriptValidationError(
            "workflow script `main` must be declared `async def main(input)` "
            "(found a plain `def main`, not `async`)",
        )
    if not _main_accepts_single_input(main):
        raise WorkflowScriptValidationError(
            "workflow script `async def main` must accept the single `input` "
            "parameter (signature `async def main(input)`)",
        )

    # 4/5. AST-derived required-surface union vs. the declared surface.
    declared_tools = _declared_tool_names(tools or [])
    declared_mcp = {s.name for s in (mcp_servers or [])}

    required_tools = _extract_required_tools(tree)
    missing_tools = sorted(required_tools - declared_tools)
    if missing_tools:
        names = ", ".join(repr(n) for n in missing_tools)
        raise WorkflowScriptValidationError(
            f"workflow script calls tool(s) not in the declared tool surface: {names}. "
            "Add them to the workflow's declared `tools` (the declared surface must be a "
            "superset of the tools the script calls by string literal).",
            detail={"missing_tools": missing_tools},
        )

    # A literal agent_id participates in the union: it is covered when a declared
    # tool OR a declared mcp server of that name exists. The child agent's full
    # transitive surface is out of scope for this DB-free create-time validator
    # (documented above); literal-only participation is the implemented depth.
    required_agent_ids = _extract_required_agent_ids(tree)
    covered = declared_tools | declared_mcp
    missing_agents = sorted(required_agent_ids - covered)
    if missing_agents:
        names = ", ".join(repr(n) for n in missing_agents)
        raise WorkflowScriptValidationError(
            f"workflow script references agent(agent_id=…) not covered by the declared "
            f"surface: {names}. Declare the corresponding tool/mcp server (the declared "
            "surface must be a superset of the string-literal agent references).",
            detail={"missing_agents": missing_agents},
        )
