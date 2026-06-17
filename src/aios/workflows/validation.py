"""Create-time validation of a workflow author script (#1284, part of #777).

A workflow is authored as an inert ``script: str`` whose only contract is a
docstring, and its declared tool/server surface is authored *separately* from the
script. Historically three failure classes surfaced late — only as a *failed run*
read back from the journal — instead of at authoring time:

1. **Syntax / structure errors** — a typo, or a missing top-level
   ``async def main(input)``, was not caught until the host ``exec``'d the script
   in a run.
2. **Silent surface drift** — the run's declared ``tools`` / ``http_servers`` /
   ``mcp_servers`` must be the **union** of the script's own ``tool("…")`` calls
   and every named ``agent(agent_id="…")`` child's surface (child surface =
   ``agent ∩ run``, the #794 clamp). Under-declaring let the clamp **silently
   strip** the missing tools, and the script hit a runtime route-mismatch — not a
   load error. ``dev_pipeline.py`` documents two production incidents from exactly
   this (an omitted ``DELETE`` silently no-op'd every unlabel; an omitted ``PATCH``
   left merged issues open).

This module lifts the compile the host already does
(:func:`aios.workflows.wf_script_host._build_coroutine`) to **create time**, plus
an AST structure check and an AST-derived **validate-declared** surface check, and
raises a precise :class:`aios.errors.ValidationError` (a clean 4xx, model-visible).

**Design decision (SETTLED — validate-declared, NOT auto-derive).** The author
writes the declared surface; the validator checks the declared surface is a
**superset** of the AST-extracted *required* surface. When a ``tool(name, …)`` name
or an ``agent_id`` is **not a string literal** (computed / a variable / from
``input``), that element is **un-AST-able** and is **excluded** from the
required-surface check — the validator neither rejects on it nor tries to resolve
it. Only string-literal ``tool("…")`` names and string-literal
``agent(agent_id="…")`` references participate in the union.

Value-domain / determinism traps (failure class 3) are out of scope here (#934).
"""

from __future__ import annotations

import ast

from aios.errors import ValidationError
from aios.models.agents import ToolSpec

# The filename the host compiles under (``aios.workflows.wf_script_host``); kept
# identical so a create-time compile error reports the same ``<workflow>`` frame
# an author would see from a run traceback.
_COMPILE_FILENAME = "<workflow>"


def declared_tool_name(spec: ToolSpec) -> str:
    """The name a ``tool("…")`` call uses to reference this declared tool.

    Mirrors the run-time surface gate (:func:`aios.workflows.run_tools.gate_run_tool`,
    which keys builtins by ``type``) and the attenuation tool identity: a custom tool
    is referenced by its ``name``; an MCP toolset by its server name; every other
    (builtin) tool by its ``type`` string (``"bash"``, ``"http_request"``, …).
    """
    if spec.type == "custom":
        return spec.name or ""
    if spec.type == "mcp_toolset":
        return spec.mcp_server_name or ""
    return str(spec.type)


def _literal_str(node: ast.expr | None) -> str | None:
    """Return the value of ``node`` iff it is a string literal, else ``None``.

    A non-literal (a ``Name``, an attribute, a subscript of ``input``, an f-string,
    a call result) is **un-AST-able** and excluded from the required-surface check.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _SurfaceVisitor(ast.NodeVisitor):
    """Collect string-literal ``tool("…")`` names and ``agent(agent_id="…")`` ids.

    Only the **first positional argument** of a ``tool(...)`` call (its ``name``) and
    the **``agent_id`` keyword** of an ``agent(...)`` call participate — matching the
    injected capability signatures ``tool(name, input)`` and
    ``agent(input, *, agent_id=None, …)``. Bare-name (un-AST-able) calls are skipped.
    """

    def __init__(self) -> None:
        self.tool_names: set[str] = set()
        self.agent_ids: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # We match the *injected* builtins by their bare name only (``tool(...)`` /
        # ``agent(...)``); an attribute access (``obj.tool(...)``) is something else.
        if isinstance(func, ast.Name):
            if func.id == "tool" and node.args:
                name = _literal_str(node.args[0])
                if name is not None:
                    self.tool_names.add(name)
            elif func.id == "agent":
                for kw in node.keywords:
                    if kw.arg == "agent_id":
                        agent_id = _literal_str(kw.value)
                        if agent_id is not None:
                            self.agent_ids.add(agent_id)
        self.generic_visit(node)


class ExtractedSurface:
    """The AST-extracted required surface (literal-only) of a workflow script."""

    __slots__ = ("agent_ids", "tool_names")

    def __init__(self, tool_names: set[str], agent_ids: set[str]) -> None:
        self.tool_names = tool_names
        self.agent_ids = agent_ids


def _check_main_signature(tree: ast.Module) -> None:
    """Assert a top-level ``async def main(input)`` exists with the right shape.

    Rejected (create-time :class:`ValidationError`):
      * no top-level ``main`` at all,
      * a ``main`` that is not ``async`` (a plain ``def main``),
      * an ``async def main`` whose signature does not accept the single ``input``
        parameter (``async def main()``, ``async def main(a, b)``, a required
        second parameter, ``*args``/``**kwargs`` only, …).

    Exactly one positional parameter is required; its *name* is not enforced (the
    contract names it ``input``, but the load-bearing check is arity — a single
    ``input`` arg). A trailing default on that one parameter is fine
    (``async def main(input=None)`` still binds ``main(input)``); any second
    parameter, ``*args``, or keyword-only parameter is rejected.
    """
    async_main: ast.AsyncFunctionDef | None = None
    plain_main: ast.FunctionDef | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.AsyncFunctionDef) and stmt.name == "main":
            async_main = stmt
        elif isinstance(stmt, ast.FunctionDef) and stmt.name == "main":
            plain_main = stmt

    if async_main is None:
        if plain_main is not None:
            raise ValidationError(
                "workflow script defines `main` but it is not async; the entry point "
                "must be `async def main(input)`",
                detail={"reason": "main_not_async"},
            )
        raise ValidationError(
            "workflow script must define a top-level `async def main(input)` entry point",
            detail={"reason": "missing_main"},
        )

    args = async_main.args
    # The single accepted parameter may arrive positional-or-keyword or
    # positional-only; ``input`` is conventionally positional-or-keyword.
    positional = list(args.posonlyargs) + list(args.args)
    n_positional = len(positional)
    # ``main(input)`` must bind exactly one positional argument: reject zero
    # parameters, and reject a second *required* positional parameter. A second
    # parameter that carries a default would still bind ``main(input)``, but the
    # contract is a single ``input`` — keep it strict and reject extra params.
    if n_positional != 1 or args.vararg is not None or args.kwonlyargs:
        raise ValidationError(
            "workflow `async def main` must accept exactly the single `input` "
            f"parameter (`async def main(input)`); got signature with {n_positional} "
            "positional parameter(s)"
            + (" plus *args" if args.vararg is not None else "")
            + (" plus keyword-only parameter(s)" if args.kwonlyargs else ""),
            detail={"reason": "bad_main_signature"},
        )


def extract_required_surface(script: str) -> ExtractedSurface:
    """Compile-check ``script``, assert ``async def main(input)``, and AST-extract
    the literal-only required surface.

    Raises :class:`aios.errors.ValidationError` (a create-time 422, model-visible)
    for a compile/syntax failure, a missing ``main``, or a mis-signatured ``main``.
    Returns the :class:`ExtractedSurface` (string-literal ``tool`` names + ``agent_id``
    references) for the caller's surface-coverage check.
    """
    # 1. Compile — the host already does exactly this (``dont_inherit=True``) in the
    #    exec path; lift it here and surface a precise, line/offset-bearing error.
    try:
        compile(script, _COMPILE_FILENAME, "exec", dont_inherit=True)
    except SyntaxError as exc:
        where = ""
        if exc.lineno is not None:
            where = f" (line {exc.lineno}"
            if exc.offset is not None:
                where += f", offset {exc.offset}"
            where += ")"
        raise ValidationError(
            f"workflow script failed to compile: {exc.msg}{where}",
            detail={
                "reason": "compile_error",
                "lineno": exc.lineno,
                "offset": exc.offset,
                "msg": exc.msg,
            },
        ) from exc

    # ``ast.parse`` cannot raise here — ``compile`` above already accepted the source.
    tree = ast.parse(script, filename=_COMPILE_FILENAME)

    # 2. Assert the entry point shape.
    _check_main_signature(tree)

    # 3. AST-extract the literal-only required surface.
    visitor = _SurfaceVisitor()
    visitor.visit(tree)
    return ExtractedSurface(visitor.tool_names, visitor.agent_ids)


def check_tool_surface(
    required_tool_names: set[str], declared_tools: list[ToolSpec] | None
) -> None:
    """Raise :class:`ValidationError` if a literal ``tool("X")`` name is not declared.

    Validate-declared: the declared tool surface must be a **superset** of the
    AST-extracted required tool names. A missing name is named in the error.
    """
    declared_names = {declared_tool_name(t) for t in (declared_tools or [])}
    missing = sorted(name for name in required_tool_names if name not in declared_names)
    if missing:
        raise ValidationError(
            "workflow script calls tool(s) not present in the declared `tools` surface: "
            + ", ".join(repr(m) for m in missing)
            + " — add them to the workflow's declared tools (the surface must be a "
            "superset of the tools the script calls)",
            detail={"reason": "undeclared_tools", "missing_tools": missing},
        )


def validate_workflow_script(
    script: str,
    *,
    tools: list[ToolSpec] | None = None,
) -> ExtractedSurface:
    """The single create-/update-time validator entry point (structure + tool surface).

    Compile + ``async def main(input)`` assertion + literal-only tool-surface
    superset check. Returns the :class:`ExtractedSurface` so a caller with DB access
    (the service path) can additionally resolve and check string-literal ``agent_id``
    references against the declared surface (see
    :func:`aios.services.workflows._check_agent_surface`).
    """
    extracted = extract_required_surface(script)
    check_tool_surface(extracted.tool_names, tools)
    return extracted
