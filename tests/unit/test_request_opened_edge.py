"""Structural / writer-provenance guards for the ``request_opened`` edge (#1123).

The trust in the request edge derives from its writer being **service-code-only**:
``append_request_opened`` is the single constructor of a ``request_opened``
lifecycle frame, it is called from exactly the launch-path creation sites, and no
harness/model-facing append path can produce it. These checks parse the source AST
(in the spirit of ``tests/unit/test_append_event_structure.py``) so they need no DB
and run in the fast unit tier; the DB-backed behavioral coverage lives in
``tests/integration/test_request_opened_edge.py``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable

from aios.db.queries import sessions as sessions_q
from aios.services import sessions as sessions_svc


def _src(obj: Callable[..., object]) -> str:
    return textwrap.dedent(inspect.getsource(obj))


def _func_def(obj: Callable[..., object], name: str) -> ast.AsyncFunctionDef:
    tree = ast.parse(_src(obj))
    func = tree.body[0]
    assert isinstance(func, ast.AsyncFunctionDef), f"expected {name} to be an async def"
    return func


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_append_request_opened_is_single_constructor_of_the_frame() -> None:
    """The ``request_opened`` data literal is built in exactly one place.

    Scan every ``db/queries`` and ``services`` source module for a dict literal with
    an ``"event": "request_opened"`` key. The only one must live inside
    ``append_request_opened`` — so no other code path can forge the frame.
    """
    from aios import services as services_pkg
    from aios.db import queries as queries_pkg

    offenders: list[str] = []
    for pkg in (queries_pkg, services_pkg):
        pkg_dir = pkg.__path__[0]
        import pathlib

        for path in pathlib.Path(pkg_dir).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                for k, v in zip(node.keys, node.values, strict=False):
                    if (
                        isinstance(k, ast.Constant)
                        and k.value == "event"
                        and isinstance(v, ast.Constant)
                        and v.value == "request_opened"
                    ):
                        offenders.append(f"{path}:{node.lineno}")
    assert len(offenders) == 1, f"request_opened frame built in >1 place: {offenders}"
    # And that one place is append_request_opened, in db/queries/sessions.py.
    assert offenders[0].endswith("sessions.py:" + offenders[0].split(":")[-1])
    func_src = _src(sessions_q.append_request_opened)
    assert '"event": "request_opened"' in func_src


def test_append_request_opened_emits_lifecycle_kind() -> None:
    """The frame is appended with ``kind='lifecycle'`` — never ``message``."""
    func = _func_def(sessions_q.append_request_opened, "append_request_opened")
    append_calls = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and _call_name(node) == "append_event"
    ]
    assert len(append_calls) == 1
    kinds = [
        kw.value.value
        for call in append_calls
        for kw in call.keywords
        if kw.arg == "kind" and isinstance(kw.value, ast.Constant)
    ]
    assert kinds == ["lifecycle"]


def test_create_child_session_calls_append_request_opened_after_first_spawn_guard() -> None:
    """``create_child_session`` opens the edge only on a real insert.

    The ``append_request_opened`` call must appear textually after the
    ``if child is None: return False`` first-spawn guard, so a replayed wake (which
    early-returns at the guard) never re-opens the edge — exactly-once.
    """
    func = _func_def(sessions_svc.create_child_session, "create_child_session")

    # Find the line of `if child is None: return False`.
    guard_line: int | None = None
    for node in ast.walk(func):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "child"
        ):
            guard_line = node.lineno
    assert guard_line is not None, "expected an `if child is None` first-spawn guard"

    opened_calls = [
        node.lineno
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and _call_name(node) == "append_request_opened"
    ]
    assert len(opened_calls) == 1, "create_child_session must open the edge exactly once"
    assert opened_calls[0] > guard_line, "edge must be opened only after the first-spawn guard"


def test_create_child_session_opens_edge_inside_transaction() -> None:
    """The edge write shares the servicer-creation transaction (rollback → no edge)."""
    func = _func_def(sessions_svc.create_child_session, "create_child_session")
    txn = next(node for node in ast.walk(func) if isinstance(node, ast.AsyncWith))
    end = max(
        (n.lineno for n in ast.walk(txn) if hasattr(n, "lineno")),
        default=txn.lineno,
    )
    opened = [
        node.lineno
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and _call_name(node) == "append_request_opened"
    ]
    assert opened, "expected an append_request_opened call"
    assert all(txn.lineno < ln <= end for ln in opened), "edge must be inside the txn block"


def test_harness_append_path_has_no_route_to_request_opened() -> None:
    """No model/harness-facing module constructs a ``request_opened`` frame.

    A model-authored event flows through the harness append paths; none of them may
    build the lifecycle frame. Scan the harness + tools packages for the literal.
    """
    import pathlib

    from aios import harness as harness_pkg
    from aios import tools as tools_pkg

    offenders: list[str] = []
    for pkg in (harness_pkg, tools_pkg):
        pkg_dir = pkg.__path__[0]
        for path in pathlib.Path(pkg_dir).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                for k, v in zip(node.keys, node.values, strict=False):
                    if (
                        isinstance(k, ast.Constant)
                        and k.value == "event"
                        and isinstance(v, ast.Constant)
                        and v.value == "request_opened"
                    ):
                        offenders.append(f"{path}:{node.lineno}")
    assert offenders == [], f"harness/tools must not author request_opened: {offenders}"


def _sql_literals(obj: Callable[..., object]) -> str:
    """Concatenate every string-constant literal in ``obj``'s body — i.e. the SQL
    fragments — EXCLUDING the docstring, so a docstring mention can't shadow the
    real query text."""
    func = _func_def(obj, getattr(obj, "__name__", "?"))
    body = func.body[1:] if ast.get_docstring(func) is not None else func.body
    parts: list[str] = []
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                parts.append(node.value)
    return "\n".join(parts)


def test_get_open_request_ids_reads_request_opened_not_metadata_blob() -> None:
    """The open-request reader derives from the trusted ``request_opened`` frame.

    Inspects the SQL fragments only (docstring excluded), so the assertion fails if
    the query reverts to the forgeable ``metadata.request`` blob.
    """
    sql = _sql_literals(sessions_q.get_open_request_ids)
    assert "request_opened" in sql, "open set must read the request_opened frame"
    # It must MINUS the answered set.
    assert "request_response" in sql
    # It must NOT read the asked set off the forgeable metadata blob anymore.
    assert "'metadata'->'request'" not in sql
