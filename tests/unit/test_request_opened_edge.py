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
    """The spine's NewSession arm opens the edge only on a real insert.

    The ``append_request_opened`` call must appear textually after the
    ``if child is None: return False`` first-spawn guard, so a replayed wake (which
    early-returns at the guard) never re-opens the edge — exactly-once. Since #1197
    factored the three near-copy creation paths onto the private ``stimulate``
    spine, the edge writer lives in ``_stimulate_new_session`` (the NewSession arm
    ``create_child_session`` is now a thin caller of).
    """
    func = _func_def(sessions_svc._stimulate_new_session, "_stimulate_new_session")

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
    """The edge write shares the servicer-creation transaction (rollback → no edge).

    The edge writer lives in the spine's NewSession arm (#1197).
    """
    func = _func_def(sessions_svc._stimulate_new_session, "_stimulate_new_session")
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


# ─── #1197: the `awaited` bit + the `Ask | Tell` discriminated union ──────────


def test_open_request_ids_filters_awaited_true() -> None:
    """The open-set reader is one of the awaited-triad readers: it must filter
    ``awaited=true`` (absent → true), so a ``Tell(NewSession)``'s unawaited edge is
    excluded and a fire-and-forget spawn never wrongly owes a response."""
    sql = _sql_literals(sessions_q.get_open_request_ids)
    assert "awaited" in sql, "open set must filter on the awaited bit (#1197)"
    # Absent ⇒ awaited=true (additive/legacy): the filter must COALESCE to TRUE.
    assert "COALESCE" in sql.upper()


def test_append_request_opened_signature_has_awaited() -> None:
    """``append_request_opened`` takes an explicit ``awaited`` field (default true)
    — never inferred from request_id-presence, so the union discipline holds."""
    import inspect

    sig = inspect.signature(sessions_q.append_request_opened)
    assert "awaited" in sig.parameters
    assert sig.parameters["awaited"].default is True


def test_stimulate_union_makes_illegal_combinations_unrepresentable() -> None:
    """The ``Ask | Tell`` union is a TYPE at the spine boundary, not a runtime
    guard: illegal combinations have no constructor.

    * no ``AskExistingSession`` carries ``vault_ids`` (a non-creating target binds
      none);
    * no ``Tell*`` arm carries ``output_schema`` (a Tell owes no response).
    """
    import dataclasses

    ask_existing_fields = {f.name for f in dataclasses.fields(sessions_svc.AskExistingSession)}
    assert "vault_ids" not in ask_existing_fields

    for tell_cls in (sessions_svc.TellNewSession, sessions_svc.TellExistingSession):
        tell_fields = {f.name for f in dataclasses.fields(tell_cls)}
        assert "output_schema" not in tell_fields, f"{tell_cls.__name__} must not own output_schema"


def test_stimulate_is_not_model_callable() -> None:
    """The spine is service-internal: it is NOT registered as a model-facing tool.

    A weak structural proxy for the mechanism/policy split — ``stimulate`` is a
    plain service function, not decorated/registered into the tool registry.
    """
    # The tool registry never imports `stimulate`; the public surface is the
    # Ask|Tell type, not a `deliver()` bool. Assert no `deliver` public function
    # leaked onto the service module.
    assert not hasattr(sessions_svc, "deliver"), "no public deliver(); spine is private"


# ─── #1413: get_open_obligations + the additive summary on the frame ──────────


def test_get_open_obligations_reads_request_opened_not_metadata_blob() -> None:
    """The obligations reader derives from the same trusted ``request_opened`` frame
    as ``get_open_request_ids`` — never the forgeable ``metadata.request`` blob."""
    sql = _sql_literals(sessions_q.get_open_obligations)
    assert "request_opened" in sql, "obligations must read the request_opened frame"
    assert "request_response" in sql, "obligations must MINUS the answered set"
    assert "'metadata'->'request'" not in sql, "must not read the forgeable metadata blob"


def test_get_open_obligations_filters_awaited_and_orders_oldest_first() -> None:
    """Lockstep with ``get_open_request_ids``: awaited filter (COALESCE TRUE) and
    oldest-first ordering so the rendered block is deterministic."""
    sql = _sql_literals(sessions_q.get_open_obligations)
    assert "awaited" in sql, "obligations must filter the awaited bit (#1197 triad)"
    assert "COALESCE" in sql.upper(), "absent awaited ⇒ TRUE"
    assert "ORDER BY req.seq ASC" in sql, "oldest-first like get_open_request_ids"


def test_get_open_obligations_projects_caller_kind_opened_at_summary() -> None:
    """The reader projects the per-obligation fields the block + read-model need:
    caller kind (trusted frame), opened_at (age), and the additive summary."""
    sql = _sql_literals(sessions_q.get_open_obligations)
    assert "'caller'->>'kind'" in sql, "caller_kind from the trusted caller frame"
    assert "created_at" in sql, "opened_at = the edge's created_at (for age)"
    assert "'summary'" in sql, "summary preview projection"


def _writer_passes_summary(obj: Callable[..., object]) -> bool:
    """True iff the function passes a ``summary=`` keyword to ``append_request_opened``."""
    func = _func_def(obj, getattr(obj, "__name__", "?"))
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and _call_name(node) == "append_request_opened"
            and any(kw.arg == "summary" for kw in node.keywords)
        ):
            return True
    return False


def test_both_writers_pass_summary_to_append_request_opened() -> None:
    """Both edge writers — the workflow-child (``_stimulate_new_session``) and the
    peer/api-invoke (``_stimulate_existing_ask``) — feed the additive #1413 summary
    so the obligations block has a human-readable preview (Issue C's set_goal
    inherits it free through ``_stimulate_existing_ask``)."""
    assert _writer_passes_summary(sessions_svc._stimulate_new_session)
    assert _writer_passes_summary(sessions_svc._stimulate_existing_ask)


def test_append_request_opened_summary_is_additive() -> None:
    """The summary is written only when present (absent ⇒ no key ⇒ id-only render,
    no migration). The frame data dict is built conditionally, not unconditionally."""
    func_src = _src(sessions_q.append_request_opened)
    assert 'data["summary"] = summary' in func_src
    assert "if summary is not None" in func_src
