"""Regression test for #1734.

``find_tool_result_event`` (``db/queries/events.py``) runs on *every*
tool-result append while the session row is locked ``FOR UPDATE``
(``harness/tool_dispatch.py``), so its WHERE clause must hit the partial
index ``events_tool_result_idx``. That index (migrations 0023 / 0097) is
predicated on the normalized ``role`` column, not the ``data->>'role'``
JSONB expression — the two are not interchangeable as far as the planner
is concerned, so filtering on the JSONB expression makes Postgres fall
back to a full per-session sequential scan under the lock.

This pins the query text to the column form so a future edit can't
silently regress back to the JSONB expression (and lose the index) without
tripping a test — a cheap structural floor since exercising the actual
query plan needs a live Postgres (see the migration-0097 EXPLAIN-adjacent
integration tests for that side of the guarantee).
"""

from __future__ import annotations

import inspect

from aios.db.queries import events as events_queries


def test_find_tool_result_event_filters_on_role_column() -> None:
    source = inspect.getsource(events_queries.find_tool_result_event)

    assert "data->>'role'" not in source, (
        "find_tool_result_event must not filter on the data->>'role' JSONB "
        "expression: it does not match the events_tool_result_idx partial "
        "index (predicated on the role column, migrations 0023/0097), so "
        "Postgres falls back to a full per-session scan under the sessions "
        "row lock on every tool-result append (#1734)."
    )
    assert "role = 'tool'" in source, (
        "find_tool_result_event must filter tool-role events via the "
        "normalized role column so the query hits events_tool_result_idx."
    )
