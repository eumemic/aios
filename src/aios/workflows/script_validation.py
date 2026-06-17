"""Create-time validation of a workflow's author ``script`` (#1285).

A workflow is authored out-of-band as an inert ``script: str`` whose only contract
is a docstring. Without create-time checking, three failure classes surface late —
only as a *failed run* read back from the journal — instead of as an authoring-time
error. This module lifts two of them (#1285 covers classes 1 and 2; the value-domain
class is #934) to ``create_workflow`` / ``update_workflow``:

1. **Syntax / structure** — the host already does
   ``compile(source, "<workflow>", "exec", dont_inherit=True)`` in the exec path
   (``wf_script_host._build_coroutine``); :func:`validate_workflow_script` lifts that
   compile to create time and surfaces a precise compile/syntax error.
2. **Surface drift** — the run's declared ``tools`` / ``http_servers`` /
   ``mcp_servers`` are authored *separately* from the script, but must be a superset
   of the surface the script actually exercises. We **validate-declared** (NOT
   auto-derive, the settled fork): AST-extract every **string-literal**
   ``tool("<name>", …)`` call name and every string-literal
   ``agent(agent_id="<id>")`` reference, compute the required-surface union, and
   assert the declared surface covers it — naming the specific missing element.

**Literal-only extraction (settled design).** When a ``tool(name, …)`` name or an
``agent_id`` is **not a string literal** (computed / a variable / from ``input``),
that element is *un-AST-able* and is **excluded** from the required-surface check —
the validator neither rejects on it nor tries to resolve it. Only string-literal
names participate in the union, so a dynamic ``tool(name_var, …)`` never causes a
false rejection.

**Isolation note.** This module imports only ``ast`` (stdlib) +
:mod:`aios.errors`. It is invoked from the credential-bearing service layer
(``aios.services.workflows``), NOT from the credential-free script-host child, so
it carries no new isolation risk for the host.

**Agent-reference depth (documented per the issue).** A ``agent(agent_id="A")``
child runs over ``agent ∩ run`` — the #794 clamp — so a named child can only ever
receive a *subset* of the run's already-declared surface; it never adds a new
tool/server requirement to the run. We therefore extract and surface literal
``agent_id`` references (excluding non-literal ones), and they participate in the
analysis, but per the workflow surface model they impose no *additional* create-time
requirement that DB-free static analysis could check: resolving the named child
agent's own tool surface to assert run-coverage requires a DB lookup and is out of
scope for this change (the issue explicitly permits documenting this depth). The
within-scope surface gate is the script's own literal ``tool("…")`` calls.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field

from aios.errors import ValidationError

# The author entry point the host execs: ``async def main(input)`` (one positional
# parameter named ``input``). Kept in sync with ``wf_script_host._build_coroutine``.
_MAIN_NAME = "main"
_MAIN_PARAM = "input"


class WorkflowScriptValidationError(ValidationError):
    """A workflow ``script`` failed create-time validation (#1285).

    A ``422`` ``ValidationError`` subtype so the HTTP layer renders it like any other
    request-shape rejection and the model-path dispatch (``_classify_tool_error``)
    returns it cleanly to the model (a 4xx author error, not a 5xx). The message is
    actionable: it names the compile failure, the missing/mis-signatured ``main``, or
    the specific under-declared tool/server.
    """

    error_type = "workflow_script_validation_error"


@dataclass
class _ScriptRequirements:
    """The AST-extracted, literal-only required surface of a workflow script."""

    tool_names: set[str] = field(default_factory=set)
    agent_ids: set[str] = field(default_factory=set)


def _literal_str(node: ast.expr | None) -> str | None:
    """Return the string value of a string-literal node, else ``None`` (un-AST-able)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _find_main(tree: ast.Module) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    """Return the LAST top-level ``main`` def (async or plain), mirroring exec rebind
    semantics where a later ``def main`` shadows an earlier one in the namespace."""
    found: ast.AsyncFunctionDef | ast.FunctionDef | None = None
    for stmt in tree.body:
        if isinstance(stmt, (ast.AsyncFunctionDef, ast.FunctionDef)) and stmt.name == _MAIN_NAME:
            found = stmt
    return found


def _assert_main_signature(fn: ast.AsyncFunctionDef | ast.FunctionDef) -> None:
    """Assert the top-level ``main`` is ``async def main(input)`` (single ``input`` arg)."""
    if not isinstance(fn, ast.AsyncFunctionDef):
        raise WorkflowScriptValidationError(
            "workflow script's top-level `main` must be `async def main(input)` — "
            "found a plain `def main` (not async)"
        )
    args = fn.args
    # Exactly one parameter named ``input``: no extra positionals, no *args/**kwargs,
    # no keyword-only params, no defaults that would let a 0-arg call through.
    positional = args.posonlyargs + args.args
    bad_shape = (
        len(positional) != 1
        or positional[0].arg != _MAIN_PARAM
        or args.vararg is not None
        or args.kwarg is not None
        or len(args.kwonlyargs) > 0
    )
    if bad_shape:
        got = ", ".join(
            [a.arg for a in args.posonlyargs]
            + [a.arg for a in args.args]
            + (["*" + args.vararg.arg] if args.vararg else [])
            + [a.arg for a in args.kwonlyargs]
            + (["**" + args.kwarg.arg] if args.kwarg else [])
        )
        raise WorkflowScriptValidationError(
            "workflow script's `main` must accept exactly one `input` parameter "
            f"(`async def main(input)`) — found signature `main({got})`"
        )


def _extract_requirements(tree: ast.Module) -> _ScriptRequirements:
    """Walk the AST for string-literal ``tool("X", …)`` names and ``agent(agent_id="A")``
    references. Non-literal names are *un-AST-able* and deliberately skipped."""
    reqs = _ScriptRequirements()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name):
            continue
        if func.id == "tool":
            # tool(name, input): the first positional arg is the tool name.
            if node.args:
                name = _literal_str(node.args[0])
                if name is not None:
                    reqs.tool_names.add(name)
        elif func.id == "agent":
            # agent(input, *, agent_id="A", …): agent_id is keyword-only.
            for kw in node.keywords:
                if kw.arg == "agent_id":
                    agent_id = _literal_str(kw.value)
                    if agent_id is not None:
                        reqs.agent_ids.add(agent_id)
    return reqs


def _declared_tool_names(tools: Sequence[object] | None) -> set[str]:
    """The canonical tool-name set of a declared surface, matching run-time resolution
    (``resolve_permission`` / the run-tool gate): a custom tool's name is its ``name``;
    a built-in's is its ``type``. MCP toolsets declare no bare ``tool("…")`` name."""
    names: set[str] = set()
    for spec in tools or []:
        type_ = getattr(spec, "type", None)
        if type_ == "custom":
            name = getattr(spec, "name", None)
            if isinstance(name, str):
                names.add(name)
        elif isinstance(type_, str) and type_ != "mcp_toolset":
            names.add(type_)
    return names


def validate_workflow_script(
    script: str,
    *,
    tools: Sequence[object] | None = None,
    mcp_servers: Sequence[object] | None = None,
    http_servers: Sequence[object] | None = None,
) -> None:
    """Validate a workflow ``script`` at author time (create + update), raising
    :class:`WorkflowScriptValidationError` with a precise, actionable message.

    Checks, in order:

    1. **Compile** the script with the host's exact flags. A ``SyntaxError`` is
       surfaced as a compile failure with line/offset where available.
    2. **Assert** a top-level ``async def main(input)`` exists.
    3. **Assert** its signature is exactly the single ``input`` parameter.
    4. **AST-extract** literal ``tool("…")`` names and assert the declared ``tools``
       cover them, naming any missing tool.

    Non-literal tool names / agent_ids are excluded from the surface check (the
    settled literal-only extraction). ``mcp_servers`` / ``http_servers`` are accepted
    for parity with the declared-surface signature; the script's own ``tool("…")``
    union is the within-scope create-time gate (see the module docstring on agent
    depth).
    """
    # 1. Compile — lift the host's exec-path compile to create time.
    try:
        compile(script, "<workflow>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        where = ""
        if exc.lineno is not None:
            where = f" (line {exc.lineno}"
            if exc.offset is not None:
                where += f", offset {exc.offset}"
            where += ")"
        raise WorkflowScriptValidationError(
            f"workflow script failed to compile: {exc.msg}{where}"
        ) from exc

    # The compile above guarantees ``ast.parse`` cannot raise on the same source.
    tree = ast.parse(script)

    # 2. + 3. Assert a top-level ``async def main(input)`` of the right signature.
    main_fn = _find_main(tree)
    if main_fn is None:
        raise WorkflowScriptValidationError(
            "workflow script must define a top-level `async def main(input)` "
            "(no `main` found)"
        )
    _assert_main_signature(main_fn)

    # 4. Validate the declared tool surface is a superset of the literal ``tool("…")``
    #    union. Name every missing tool (sorted for a deterministic message).
    reqs = _extract_requirements(tree)
    declared = _declared_tool_names(tools)
    missing = sorted(reqs.tool_names - declared)
    if missing:
        named = ", ".join(repr(m) for m in missing)
        raise WorkflowScriptValidationError(
            "workflow script calls tool(s) not present in the declared tool surface: "
            f"{named}. Declare them in `tools` (the declared surface must be a superset "
            "of the script's literal tool(\"…\") calls)."
        )
