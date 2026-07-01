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
    def test_transaction_opens_after_precompute(self) -> None:
        """The ``async with conn.transaction()`` must appear AFTER the
        pre-transaction compute.

        Since #991 ``append_event`` delegates the token-delta + parent-lookup
        compute to ``precompute_event_append`` (so the two tool-result appenders
        can run it OUTSIDE their outer lock).  The fallback call to it — for
        every non-tool-result caller, when ``precomputed=None`` — must still sit
        strictly above the row-lock transaction.
        """
        func = _append_event_def()
        tx_lineno = _transaction_lineno(func)

        linenos = _call_linenos(func, "precompute_event_append")
        assert linenos, "expected at least one call to precompute_event_append in append_event"
        for lineno in linenos:
            assert lineno < tx_lineno, (
                f"precompute_event_append (line {lineno}) must be called before the "
                f"transaction opens (line {tx_lineno}), not inside the row lock"
            )

    def test_precompute_helper_runs_token_and_lookup_calls_no_io_pre_lock(self) -> None:
        """The extracted ``precompute_event_append`` is where the token delta,
        parent-channel lookup, and focal read live — verify each helper is
        present there so a rename/removal fails loudly rather than silently."""
        src = textwrap.dedent(inspect.getsource(events_mod.precompute_event_append))
        tree = ast.parse(src)
        func = tree.body[0]
        assert isinstance(func, ast.AsyncFunctionDef)
        for name in (
            "_event_token_delta",
            "_lookup_tool_parent_channel",
            "get_session_focal_channel",
        ):
            assert _call_linenos(func, name), (
                f"expected at least one call to {name} in precompute_event_append"
            )

    def test_only_cumulative_query_between_update_and_insert(self) -> None:
        """In the region between the seq-allocating UPDATE and the INSERT, the
        only awaited query helper is ``_latest_cumulative_state`` — no
        tokenizer, no channel lookup, no focal read.

        (``_latest_cumulative_state`` is the single-index-seek read of the
        prior message row's running counters — ``cumulative_tokens`` plus the
        ``cumulative_messages`` count and per-class mass added in issue #1657;
        it replaced the narrower ``_latest_cumulative_tokens`` fetch here so
        every running sum advances off one seek, still under the row lock.)"""
        func = _append_event_def()

        update_lineno = _str_literal_lineno(func, "UPDATE sessions ")
        insert_lineno = _str_literal_lineno(func, "INSERT INTO events ")
        assert update_lineno < insert_lineno

        def in_region(lineno: int) -> bool:
            return update_lineno < lineno < insert_lineno

        # The region is correctly identified AND non-trivial: the cumulative
        # read must land inside it.
        cumulative = [n for n in _call_linenos(func, "_latest_cumulative_state") if in_region(n)]
        assert cumulative, (
            "_latest_cumulative_state must be called between the UPDATE and the INSERT"
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


# ── Caller-level guard (issue #991) ───────────────────────────────────────────
# The structural guard above AST-parses ``append_event`` IN ISOLATION — it has
# no view of the two tool-result appenders that wrap ``append_event`` inside
# their OWN outer ``SELECT ... FOR UPDATE`` dedup transaction.  Before #991 those
# callers passed the in-tx ``conn`` straight into ``append_event``, so its
# pre-transaction tokenizer pass ran as a nested savepoint UNDER the held session
# lock — defeating #862 on the ~100 KB tool-result path.  This sibling guard is
# the PRIMARY regression sentinel for lock placement: it asserts the
# ``precompute_event_append`` call sits strictly ABOVE the lock-bearing
# ``transaction()`` block, and that no tokenizer / parent-lookup helper runs
# below the lock open.  The sequential idempotency tests cannot catch a
# lock-narrowing regression (they await inline), so this AST check carries it.

from aios.harness import tool_dispatch as tool_dispatch_mod  # noqa: E402
from aios.services import sessions as sessions_mod  # noqa: E402


def _func_def(module: object, name: str) -> ast.AsyncFunctionDef:
    src = textwrap.dedent(inspect.getsource(getattr(module, name)))
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.AsyncFunctionDef), f"expected {name} to be an async def"
    return func


def _lock_block_lineno(func: ast.AsyncFunctionDef) -> int:
    """Line number of the LOCK-bearing ``async with`` block in a tool appender.

    The lock block is the ``async with`` whose body acquires the session row
    lock — either a ``SELECT ... FOR UPDATE`` execute (``_append_tool_result_event``)
    or a ``lock_active_session_for_update`` call (``append_tool_result``).  The
    cold-path precompute's own ``async with pool.acquire()`` block must NOT match
    (it acquires no lock), so we key on the lock marker inside the body.
    """
    candidates: list[int] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.AsyncWith):
            continue
        body_src = ast.dump(node)
        if "FOR UPDATE" in _async_with_str_literals(node) or _calls_within(
            node, "lock_active_session_for_update"
        ):
            candidates.append(node.lineno)
        _ = body_src
    assert candidates, "no lock-bearing `async with` block found in the tool appender"
    return min(candidates)


def _async_with_str_literals(node: ast.AsyncWith) -> str:
    """All string-literal constants anywhere inside an ``async with`` body."""
    parts: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            parts.append(sub.value)
    return "\n".join(parts)


def _calls_within(node: ast.AST, name: str) -> bool:
    return any(isinstance(sub, ast.Call) and _call_name(sub) == name for sub in ast.walk(node))


class TestNoInLockComputeInToolAppenders:
    """Issue #991: the two tool-result appenders must precompute the token delta
    (and cold-path parent channel) BEFORE acquiring the outer session-row lock,
    not under it."""

    APPENDERS = (
        (tool_dispatch_mod, "_append_tool_result_event"),
        (sessions_mod, "append_tool_result"),
    )

    def test_precompute_call_is_above_the_lock_block(self) -> None:
        for module, fname in self.APPENDERS:
            func = _func_def(module, fname)
            precompute_linenos = _call_linenos(func, "precompute_event_append")
            assert precompute_linenos, (
                f"{fname} must call precompute_event_append (pre-lock token compute)"
            )
            lock_lineno = _lock_block_lineno(func)
            for lineno in precompute_linenos:
                assert lineno < lock_lineno, (
                    f"{fname}: precompute_event_append (line {lineno}) must run BEFORE "
                    f"the session lock block (line {lock_lineno}), not under it"
                )

    def test_no_tokenizer_or_parent_lookup_below_the_lock(self) -> None:
        for module, fname in self.APPENDERS:
            func = _func_def(module, fname)
            lock_lineno = _lock_block_lineno(func)
            for forbidden in (
                "approx_tokens",
                "render_user_event",
                "_event_token_delta",
                "_lookup_tool_parent_channel",
                "precompute_event_append",
            ):
                offenders = [n for n in _call_linenos(func, forbidden) if n > lock_lineno]
                assert not offenders, (
                    f"{fname}: {forbidden} (line(s) {offenders}) must not run below the "
                    f"lock open (line {lock_lineno}) — it belongs in the pre-lock precompute"
                )
