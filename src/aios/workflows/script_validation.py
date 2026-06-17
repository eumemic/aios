"""Create-time validation of a workflow ``script`` (#1285, bakeoff of #1221).

A workflow is authored out-of-band as an inert ``script: str``. Before this module
the *only* contract checked at author time was the pydantic length/shape of the
field — every structural and surface-drift failure surfaced **late**, only as a
failed *run* read back off the journal:

1. **Syntax / structure** — a typo, or a missing top-level ``async def main(input)``,
   was not caught until the host ``exec``'d the script inside a run.
2. **Silent surface drift** — the declared ``tools`` / ``http_servers`` /
   ``mcp_servers`` are authored *separately* from the script but must cover the
   union of the script's own ``tool("…")`` calls (and its named ``agent(agent_id=…)``
   children — resolved in the service layer, see ``services.workflows``). An
   under-declared surface is **silently clamped** at run launch (#794), so the
   script hits a runtime route-mismatch, not a load error. ``dev_pipeline.py``
   records two production incidents from exactly this (an omitted ``DELETE`` that
   silently no-op'd every unlabel; an omitted ``PATCH`` that left merged issues open).

This module lifts the host's own ``compile(...)`` to create time and adds two AST
checks, so all of class (1) and the *script-local* half of class (2) are caught as
an **authoring-time** :class:`ValidationError` (a 4xx — the tool dispatch layer turns
it into a clean, model-visible result) instead of a failed run.

**Design (SETTLED — #1285): validate-declared, literal-only.** The author writes the
declared surface; the validator checks it is a **superset** of the AST-extracted
*required* surface. A ``tool(name, …)`` whose name is **not a string literal**
(computed / a variable / from ``input``) is **un-AST-able** and is **excluded** from
the required-surface check — the validator neither rejects on it nor attempts to
resolve it. Only string-literal names participate.
"""

from __future__ import annotations

import ast

from aios.errors import ValidationError
from aios.models.agents import ToolSpec

__all__ = [
    "declared_tool_names",
    "extract_required_agent_ids",
    "validate_workflow_script",
]


def declared_tool_names(tools: list[ToolSpec]) -> set[str]:
    """The set of names a ``tool("…")`` call may legitimately reference.

    The run-tool gate (:func:`aios.workflows.run_tools.gate_run_tool`) matches a
    call against ``t.type`` (the builtin tool name, e.g. ``"bash"``,
    ``"http_request"``). A ``custom`` tool is invoked by its ``t.name``. We accept
    **either** so the literal-extraction check stays permissive (never a false
    rejection) while still catching a flatly omitted tool. Disabled tools are
    included: an author may intentionally ship a disabled tool the script names —
    it resolves to a recoverable ``{"error": …}`` at run time, not a load failure,
    so it is not the create-time concern this validator guards.
    """
    names: set[str] = set()
    for spec in tools:
        names.add(spec.type)
        if spec.name is not None:
            names.add(spec.name)
    return names


def _main_def(tree: ast.Module) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    """The LAST top-level ``def``/``async def`` named ``main`` (last binding wins,
    mirroring ``exec`` namespace semantics where a later definition shadows an
    earlier one)."""
    found: ast.AsyncFunctionDef | ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == "main":
            found = node
    return found


def _accepts_single_input(fn: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    """True iff ``fn`` can be called as ``main(input_value)`` — exactly one positional
    argument.

    The host invokes the entry point **positionally** (``entry(input_value)`` in
    ``wf_script_host._build_coroutine``), so the check is on ARITY, not the parameter
    *name*: ``async def main(input)`` and ``async def main(i)`` are equally valid (the
    canonical name in the contract/docs is ``input``, but a differently-named single
    parameter binds the same value — and many existing workflows use ``i``). Accepts
    the single-positional shape (incl. ``def main(input, /)``) and a ``*args``
    catch-all; rejects ``main()`` (too few — the run's ``input`` would have nowhere to
    bind), ``main(a, b)`` (too many, no catch-all), and a sole parameter that is
    *keyword-only* (``def main(*, input)`` — cannot be filled positionally).
    """
    args = fn.args
    positional = args.posonlyargs + args.args
    # Exactly one positional parameter (no catch-all needed), OR no declared
    # positional params but a ``*args`` catch-all that still binds the one value.
    return (len(positional) == 1 and args.vararg is None) or (
        len(positional) == 0 and args.vararg is not None
    )


def validate_workflow_script(script: str, tools: list[ToolSpec]) -> None:
    """Validate a workflow ``script`` at author time. Raises :class:`ValidationError`
    (no return) on the first failure; returns ``None`` when the script passes the
    structural + script-local-surface checks.

    Checks, in order:

    1. **Compile** — ``compile(script, "<workflow>", "exec", dont_inherit=True)``
       (the exact host call, lifted from the run path). A failure raises naming it a
       compile/syntax error, surfacing line/offset where the ``SyntaxError`` carries
       them.
    2. **``main`` exists** — a top-level ``async def main`` (AST).
    3. **``main`` is well-signatured** — ``async`` (not a plain ``def``) and accepting
       the single ``input`` parameter.
    4. **Tool surface superset** — every string-literal ``tool("X", …)`` name is in
       the declared ``tools``; a missing one raises naming ``X``.

    The named-``agent(agent_id="A")`` half of the surface union is resolved against
    the live agent in the service layer (it needs the pool); see
    :func:`extract_required_agent_ids`.
    """
    # 1. Compile — mirror the host (wf_script_host._build_coroutine) exactly.
    try:
        compile(script, "<workflow>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        where = ""
        if exc.lineno is not None:
            where = f" (line {exc.lineno}" + (
                f", offset {exc.offset})" if exc.offset is not None else ")"
            )
        raise ValidationError(
            f"workflow script failed to compile{where}: {exc.msg}",
            detail={"lineno": exc.lineno, "offset": exc.offset, "msg": exc.msg},
        ) from exc

    tree = ast.parse(script)

    # 2. + 3. A correctly-signatured top-level ``async def main(input)``.
    main = _main_def(tree)
    if main is None:
        raise ValidationError(
            "workflow script must define a top-level `async def main(input)` "
            "(no such function found)"
        )
    if not isinstance(main, ast.AsyncFunctionDef):
        raise ValidationError(
            "workflow script `main` must be `async def main(input)` — "
            "the top-level `main` is a plain `def`, not `async`"
        )
    if not _accepts_single_input(main):
        raise ValidationError(
            "workflow script `main` must accept the single `input` parameter "
            "(`async def main(input)`)"
        )

    # 4. Tool surface superset (literal-only). A non-literal name is un-AST-able and
    #    excluded — not a violation.
    declared = declared_tool_names(tools)
    missing = sorted(name for name in _extract_literal_tool_names(tree) if name not in declared)
    if missing:
        joined = ", ".join(repr(m) for m in missing)
        raise ValidationError(
            f"workflow script calls tool(s) {joined} not in the declared tool surface; "
            "add them to the workflow's `tools` (the declared surface must cover every "
            'literal tool("…") the script calls)',
            detail={"missing_tools": missing},
        )


def _string_literal(node: ast.expr | None) -> str | None:
    """The value of ``node`` if it is a plain string-literal constant, else ``None``
    (so a computed / variable / f-string expression is treated as un-AST-able)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_literal_tool_names(tree: ast.Module) -> set[str]:
    """String-literal first-positional names of every ``tool("…", …)`` call.

    Matches a bare ``tool(...)`` call (the injected builtin; the author namespace
    binds ``tool`` directly — see ``wf_script_host.author_namespace``). A call whose
    first argument is not a string literal is **excluded** (un-AST-able), per the
    settled literal-only design.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "tool"):
            continue
        first = node.args[0] if node.args else None
        literal = _string_literal(first)
        if literal is not None:
            names.add(literal)
    return names


def extract_required_agent_ids(script: str) -> set[str]:
    """String-literal ``agent_id`` values of every ``agent(..., agent_id="A")`` call.

    The service layer resolves each to the live child agent and folds its declared
    surface into the required-surface union (the #794 ``agent ∩ run`` clamp): the
    run's declared surface must cover it, or the clamp would silently strip the
    child's tools. A non-literal ``agent_id`` (from ``input`` / a variable / omitted
    for the generic subagent) is un-AST-able and excluded.

    Pure/AST-only and importable without a pool, so callers that only need the
    structural checks (and tests) can use it standalone.
    """
    tree = ast.parse(script)
    ids: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "agent"):
            continue
        for kw in node.keywords:
            if kw.arg == "agent_id":
                literal = _string_literal(kw.value)
                if literal is not None:
                    ids.add(literal)
    return ids
