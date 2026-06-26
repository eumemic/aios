"""Read-tolerance for the RETIRED ``complete_goal``/``fail_goal`` builtins (#1562).

#1525 removed ``complete_goal``/``fail_goal`` from ``BuiltinToolType`` + the registry but shipped
neither a read shim nor a data migration. A long-lived agent whose persisted ``tools`` JSONB still
listed those builtins then failed ``ToolSpec`` validation on every wake — a pre-context-build throw
that wedged the agent into an infinite reschedule (the kedalion-ultron incident). They have NO
canonical successor (``return``/``error`` are general step verbs, not model-listed builtins), so
the entries are DROPPED, not remapped.

``load_tool_specs`` is the list-level read-tolerance choke point used by every DB read path
(agents / agent_versions / workflows / workflow_versions / wf_runs / sessions); these tests pin
the drop + order/preservation semantics, and ``test_agent_row_with_retired_builtin_hydrates_clean``
is the CI-gap test the incident calls for: test agents are born from the *current* catalog so none
carry retired builtins, which is exactly why the wedge was invisible to CI.
"""

from __future__ import annotations

import pytest

from aios.models.agents import load_tool_specs


def test_retired_builtin_entry_is_dropped() -> None:
    specs = load_tool_specs([{"type": "bash"}, {"type": "complete_goal"}, {"type": "read"}])
    assert [s.type for s in specs] == ["bash", "read"]


@pytest.mark.parametrize("retired", ["complete_goal", "fail_goal"])
def test_each_retired_builtin_is_dropped(retired: str) -> None:
    specs = load_tool_specs([{"type": retired}, {"type": "bash"}])
    assert [s.type for s in specs] == ["bash"]


def test_retired_only_list_hydrates_to_empty() -> None:
    assert load_tool_specs([{"type": "complete_goal"}, {"type": "fail_goal"}]) == []


def test_clean_list_is_untouched_and_validates() -> None:
    specs = load_tool_specs([{"type": "bash"}, {"type": "create_goal"}])
    assert [s.type for s in specs] == ["bash", "create_goal"]


def test_order_preserved_with_custom_and_mcp_entries() -> None:
    specs = load_tool_specs(
        [
            {"type": "bash"},
            {"type": "fail_goal"},
            {"type": "custom", "name": "foo", "description": "d", "input_schema": {}},
            {"type": "complete_goal"},
            {"type": "mcp_toolset", "mcp_server_name": "srv"},
        ]
    )
    assert [s.type for s in specs] == ["bash", "custom", "mcp_toolset"]


def test_retired_drop_composes_with_legacy_rename() -> None:
    """A row mixing a retired builtin and a legacy-renamed builtin: the retired one is dropped,
    the legacy one is still remapped to its canonical name (the two shims compose)."""
    specs = load_tool_specs([{"type": "complete_goal"}, {"type": "invoke_workflow"}])
    assert [s.type for s in specs] == ["call_workflow"]


def test_agent_row_with_retired_builtin_hydrates_clean() -> None:
    """CI-gap regression: an agent whose persisted ``tools`` JSONB carries a retired builtin
    hydrates without raising — the exact wedge #1525 introduced and CI missed (test agents are
    born from the current catalog, so none carry retired builtins)."""
    from datetime import UTC, datetime

    from aios.db.queries.agents import _row_to_agent

    now = datetime.now(UTC)
    row = {
        "id": "agt_wedged",
        "version": 3,
        "name": "ultron",
        "model": "anthropic/claude-opus-4-6",
        "system": "",
        # Pool reads arrive already parsed (the jsonb codec decodes), so ``parse_jsonb`` is a
        # passthrough — pass parsed Python here, exactly what ``_row_to_agent`` receives in prod.
        "tools": [{"type": "bash"}, {"type": "complete_goal"}, {"type": "fail_goal"}],
        "skills": [],
        "mcp_servers": [],
        "http_servers": [],
        "description": None,
        "metadata": {},
        "litellm_extra": {},
        "window_min": 1,
        "window_max": 10,
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
    }

    agent = _row_to_agent(row)  # type: ignore[arg-type]

    # The retired builtins are gone; the legitimate one survives. No exception = no wedge.
    assert [t.type for t in agent.tools] == ["bash"]
