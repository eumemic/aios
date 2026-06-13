"""Structural guard (issue #991): the tool-result appenders must NOT hold a
session ``FOR UPDATE`` across ``append_event``'s pre-transaction tokenizer
pass.  WP-23 (#862) moved the tokenizer out of ``append_event``'s row-lock
transaction, but the tool-result callers re-introduced an OUTER
``conn.transaction()`` + ``SELECT ... FOR UPDATE`` around the whole
``append_event`` call, so for those paths the tokenizer ran under the lock
again.  The fix routes idempotency through ``append_event``'s ``dedup`` hook,
which takes the lock only AFTER the pre-lock compute.

These structural assertions follow the AST-scan pattern of
``tests/unit/test_append_event_structure.py``: no DB, fast unit tier, immune
to docstring/comment mentions because we parse the function body.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from aios.harness import tool_dispatch as dispatch_mod
from aios.services import sessions as sessions_mod


def _func_def(func: object) -> ast.AsyncFunctionDef:
    src = textwrap.dedent(inspect.getsource(func))  # type: ignore[arg-type]
    tree = ast.parse(src)
    node = tree.body[0]
    assert isinstance(node, ast.AsyncFunctionDef), "expected an async def"
    return node


def _has_for_update_select(func: ast.AsyncFunctionDef) -> bool:
    """True if any NON-docstring string-literal in the function (or its nested
    defs) contains ``FOR UPDATE`` — the appenders used to hold a raw
    ``SELECT ... FOR UPDATE`` themselves.  Docstrings are excluded so a prose
    mention of the old structure can't mask a real regression."""
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(func)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "FOR UPDATE" in node.value
        and node.value not in docstrings
        for node in ast.walk(func)
    )


def _append_event_calls_with_dedup_kwarg(func: ast.AsyncFunctionDef) -> list[ast.Call]:
    """Every ``append_event(...)`` call that passes a ``dedup=`` keyword."""
    out: list[ast.Call] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = (
            callee.attr
            if isinstance(callee, ast.Attribute)
            else (callee.id if isinstance(callee, ast.Name) else None)
        )
        if name != "append_event":
            continue
        if any(kw.arg == "dedup" for kw in node.keywords):
            out.append(node)
    return out


class TestNoInLockTokenizerOnToolResultPaths:
    def test_harness_appender_routes_dedup_through_append_event(self) -> None:
        """``_append_tool_result_event`` (harness path) must not hold its own
        ``FOR UPDATE`` — it passes ``dedup=`` to ``append_event`` so the lock
        is taken only after the pre-lock tokenizer pass."""
        func = _func_def(dispatch_mod._append_tool_result_event)
        assert not _has_for_update_select(func), (
            "_append_tool_result_event still issues a raw SELECT ... FOR UPDATE; "
            "the lock must be taken by append_event's dedup hook, not by an "
            "outer transaction spanning the tokenizer"
        )
        assert _append_event_calls_with_dedup_kwarg(func), (
            "_append_tool_result_event must pass dedup= to append_event so the "
            "idempotency guard runs under append_event's own row lock"
        )

    def test_service_appender_routes_dedup_through_append_event(self) -> None:
        """``services.append_tool_result`` (operator/connector/ghost-repair)
        must not hold an outer ``FOR UPDATE`` across ``append_event``."""
        func = _func_def(sessions_mod.append_tool_result)
        assert not _has_for_update_select(func), (
            "append_tool_result still issues a raw SELECT ... FOR UPDATE"
        )
        assert _append_event_calls_with_dedup_kwarg(func), (
            "append_tool_result must pass dedup= to append_event"
        )

    def test_service_appender_does_not_open_outer_transaction(self) -> None:
        """No ``conn.transaction()`` block in ``append_tool_result`` — the
        only transaction is the one ``append_event`` opens internally (after
        its pre-lock compute)."""
        func = _func_def(sessions_mod.append_tool_result)
        for node in ast.walk(func):
            if not isinstance(node, ast.AsyncWith):
                continue
            for item in node.items:
                call = item.context_expr
                assert not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "transaction"
                ), (
                    "append_tool_result opens an outer conn.transaction(); the "
                    "tokenizer would run under the lock again. Route idempotency "
                    "through append_event's dedup hook instead."
                )
