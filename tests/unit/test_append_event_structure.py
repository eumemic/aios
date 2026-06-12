"""Structural guard (issue #862): neither the tokenizer pass nor the
parent-assistant channel lookup may execute inside ``append_event``'s row-lock
transaction. Between the seq-allocating UPDATE and the INSERT, the ONLY awaited
query is ``_latest_cumulative_tokens``.

In the spirit of ``TestNoEventsSubquery`` in
``tests/integration/test_session_status_scalars.py`` (which string-scans a SQL
constant), this needs no DB and runs in the fast unit tier. It parses the
``append_event`` AST rather than scanning the source string so docstring
``:func:`` cross-references and comments can never shadow the real call sites
(an ``str.index`` on the source would match the docstring mention first).
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from aios.db.queries import events as events_mod


def _append_event_def() -> ast.AsyncFunctionDef:
    src = textwrap.dedent(inspect.getsource(events_mod.append_event))
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.AsyncFunctionDef), "expected append_event to be an async def"
    return func


def _call_name(node: ast.Call) -> str | None:
    """The simple callee name of a ``Call`` node (``foo(...)`` → ``foo``,
    ``mod.foo(...)`` → ``foo``), or None for more exotic callees."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_linenos(func: ast.AsyncFunctionDef, name: str) -> list[int]:
    """Line numbers of every ``Call`` to ``name`` anywhere in ``func``."""
    return [
        node.lineno
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]


def _transaction_lineno(func: ast.AsyncFunctionDef) -> int:
    """Line number of the ``async with conn.transaction()`` block."""
    for node in ast.walk(func):
        if not isinstance(node, ast.AsyncWith):
            continue
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "transaction"
            ):
                return node.lineno
    raise AssertionError("no `async with conn.transaction()` block found in append_event")


class TestNoInLockCompute:
    def test_transaction_opens_after_all_token_and_lookup_calls(self) -> None:
        """The ``async with conn.transaction()`` must appear AFTER the last
        pre-transaction computation (token delta, parent lookup, focal read)."""
        func = _append_event_def()
        tx_lineno = _transaction_lineno(func)

        # Each compute helper that must run pre-lock is (a) present — so the
        # guard fails loudly if a helper is renamed/removed rather than
        # silently passing — and (b) every call to it is strictly above the
        # transaction block. AST line numbers are immune to docstring mentions.
        for name in (
            "_event_token_delta",
            "_lookup_tool_parent_channel",
            "get_session_focal_channel",
        ):
            linenos = _call_linenos(func, name)
            assert linenos, f"expected at least one call to {name} in append_event"
            for lineno in linenos:
                assert lineno < tx_lineno, (
                    f"{name} (line {lineno}) must be called before the transaction "
                    f"opens (line {tx_lineno}), not inside the row lock"
                )

    def test_only_cumulative_query_between_update_and_insert(self) -> None:
        """In the region between the seq-allocating UPDATE and the INSERT, the
        only awaited query helper is ``_latest_cumulative_tokens`` — no
        tokenizer, no channel lookup, no focal read."""
        func = _append_event_def()

        update_lineno = _str_literal_lineno(func, "UPDATE sessions ")
        insert_lineno = _str_literal_lineno(func, "INSERT INTO events ")
        assert update_lineno < insert_lineno

        def in_region(lineno: int) -> bool:
            return update_lineno < lineno < insert_lineno

        # The region is correctly identified AND non-trivial: the cumulative
        # read must land inside it.
        cumulative = [n for n in _call_linenos(func, "_latest_cumulative_tokens") if in_region(n)]
        assert cumulative, (
            "_latest_cumulative_tokens must be called between the UPDATE and the INSERT"
        )

        # Nothing heavier may run in that region.
        for forbidden in (
            "_lookup_tool_parent_channel",
            "approx_tokens",
            "render_user_event",
            "get_session_focal_channel",
            "_event_token_delta",
        ):
            offenders = [n for n in _call_linenos(func, forbidden) if in_region(n)]
            assert not offenders, (
                f"{forbidden} (line(s) {offenders}) must not run between the UPDATE "
                "and the INSERT (it belongs pre-transaction)"
            )


def _str_literal_lineno(func: ast.AsyncFunctionDef, prefix: str) -> int:
    """Line number of the first string-literal ``Constant`` whose value starts
    with ``prefix`` — used to anchor the UPDATE/INSERT SQL statements."""
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith(prefix)
        ):
            return node.lineno
    raise AssertionError(f"no string literal starting with {prefix!r} found in append_event")
