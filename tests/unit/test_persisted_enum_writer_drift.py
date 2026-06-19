"""Single-source guard for persisted-enum value sets (#1081).

Every persisted enum's value set must have exactly ONE editable source: the
Python ``Literal``. A forgotten paired migration used to ship a *prod-only*
runtime ``CHECK`` violation (invisible to CI). This module closes the gap from
the *writer* side: for every column whose redundant value-enum ``CHECK`` is
dropped in migration ``0111`` (the (A)/(B) rows of #1081), the write path is now
``Literal``-typed, so an illegal write is a static type error rather than a
silent landing or a prod-only constraint trip.

This is the writer-signature half of the drift guard, modelled on
``tests/unit/test_agent_tooltype_registry_drift.py``'s ``get_args(...)``
set-equality idiom. It needs no DB — pure ``inspect.signature`` + ``get_args``
set algebra. The companion ``tests/integration/test_persisted_enum_check_drift``
covers the live-``CHECK`` half (the (C) rows whose constraint is *kept*, pinned
to its ``Literal`` via ``pg_get_constraintdef``), and asserts the dropped
``CHECK``s are actually absent.

If a ``Literal`` and the writer that feeds the column diverge — e.g. someone
widens ``WfRunStatus`` but forgets to retype ``set_run_status`` — this test
fails at build time, *and* mypy flags the illegal call site. The two copies can
no longer be authored independently and pass CI.
"""

from __future__ import annotations

import sys
import typing
from typing import Any, get_args

from aios.db.queries import memory_stores as memory_store_queries
from aios.db.queries import workflows as workflow_queries
from aios.models.memory_stores import Access, ActorType
from aios.models.workflows import WfRunStatus
from aios.workflows import step as workflow_step


def _param_literal_args(func: Any, param: str) -> tuple[object, ...]:
    """``get_args`` of ``func``'s ``param`` annotation, resolving the stringized
    (``from __future__ import annotations``) hint against the *function's own*
    module globals. Returns ``()`` when the annotation is absent or not a
    ``Literal`` so the assertion message is informative.

    We resolve a single parameter rather than calling ``get_type_hints`` on the
    whole function because other params (e.g. ``conn: asyncpg.Connection[Any]``)
    carry annotations that are not subscriptable at runtime and would raise."""
    raw = func.__annotations__.get(param)
    if raw is None:
        return ()
    if isinstance(raw, str):
        module_globals = getattr(sys.modules.get(func.__module__), "__dict__", {})
        raw = eval(raw, {"typing": typing, **module_globals})
    return get_args(raw)


def test_wf_run_status_writers_are_literal_typed() -> None:
    """The (B) ``wf_runs.status`` writers must be ``WfRunStatus``-typed, not
    plain ``str`` — that is what converts ``set_run_status(conn, run_id,
    "pasued", ...)`` from a silent landing into a static type error once its
    redundant ``wf_runs_status_check`` is dropped."""
    expected = set(get_args(WfRunStatus))
    for func, param in (
        (workflow_queries.set_run_status, "status"),
        (workflow_queries.set_run_terminal, "status"),
        (workflow_step._commit_terminal_and_dispatch, "status"),
    ):
        assert set(_param_literal_args(func, param)) == expected, (
            f"{func.__qualname__}'s '{param}' must be typed WfRunStatus "
            f"(single-sourced with the Literal); got "
            f"{_param_literal_args(func, param)!r}"
        )


def test_memory_store_value_enum_writers_are_literal_typed() -> None:
    """The (B) ``memory_stores`` value-enum writers must carry the model
    ``Literal``, not plain ``str``: ``created_by_type`` flows from
    ``actor_type`` (``ActorType``) and ``session_memory_stores.access`` from
    ``access`` (``Access``)."""
    for func, param, literal in (
        (memory_store_queries.insert_memory_with_version, "actor_type", ActorType),
        (memory_store_queries.update_memory_with_version, "actor_type", ActorType),
        (memory_store_queries.delete_memory_with_version, "actor_type", ActorType),
        (memory_store_queries.redact_memory_version, "actor_type", ActorType),
        (memory_store_queries.insert_session_memory_store, "access", Access),
    ):
        assert set(_param_literal_args(func, param)) == set(get_args(literal)), (
            f"{func.__qualname__}'s '{param}' must be typed "
            f"{literal} (single-sourced with the Literal); got "
            f"{_param_literal_args(func, param)!r}"
        )
