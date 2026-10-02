"""The RETIRED ``complete_goal``/``fail_goal`` builtins now fail closed on read (#1569).

#1525 removed ``complete_goal``/``fail_goal`` from ``BuiltinToolType`` + the registry without a
migration; #1563 closed the gap with migration 0122 (which scrubbed every tool-bearing surface)
plus a temporary read shim in ``load_tool_specs`` that silently dropped the retired entries.

The 0122 contract migration has run everywhere, and the fail-closed boot-admission gate (#1575)
re-proves zero residue for these tokens on every registered surface at every boot. The read shim
is therefore removed (migrate-and-clean-break, no permanent fallback): a persisted ``tools``
array that still carries a retired builtin now RAISES instead of being silently tolerated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from aios.models.agents import BuiltinToolType, load_tool_specs


@pytest.mark.parametrize("retired", ["complete_goal", "fail_goal"])
def test_retired_builtin_entry_raises(retired: str) -> None:
    with pytest.raises(ValidationError):
        load_tool_specs([{"type": "bash"}, {"type": retired}, {"type": "read"}])


def test_retired_only_list_raises() -> None:
    with pytest.raises(ValidationError):
        load_tool_specs([{"type": "complete_goal"}, {"type": "fail_goal"}])


def test_no_retired_builtin_shim_remains() -> None:
    import aios.models.agents as agents_mod

    assert not hasattr(agents_mod, "_RETIRED_BUILTINS")


@pytest.mark.parametrize("retired", ["complete_goal", "fail_goal"])
def test_retired_builtin_not_in_current_catalog(retired: str) -> None:
    """No runtime path can mint a retired type: agents are born from the current catalog."""
    from typing import get_args

    assert retired not in get_args(BuiltinToolType)


def test_clean_list_is_untouched_and_validates() -> None:
    specs = load_tool_specs([{"type": "bash"}, {"type": "create_goal"}])
    assert [s.type for s in specs] == ["bash", "create_goal"]


def test_order_preserved_with_custom_and_mcp_entries() -> None:
    specs = load_tool_specs(
        [
            {"type": "bash"},
            {"type": "custom", "name": "foo", "description": "d", "input_schema": {}},
            {"type": "mcp_toolset", "mcp_server_name": "srv"},
        ]
    )
    assert [s.type for s in specs] == ["bash", "custom", "mcp_toolset"]


def _agent_row(tools: list[dict[str, Any]]) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": "agt_retired",
        "version": 3,
        "name": "ultron",
        "model": "anthropic/claude-opus-4-6",
        "system": "",
        # Pool reads arrive already parsed (the jsonb codec decodes).
        "tools": tools,
        "skills": [],
        "mcp_servers": [],
        "http_servers": [],
        "description": None,
        "metadata": {},
        "litellm_extra": {},
        "window_min": 1,
        "window_max": 10,
        "preempt_policy": "wait",
        "output_style": "default",
        "created_by_type": None,
        "created_by_ref": None,
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
    }


def test_agent_row_with_retired_builtin_raises() -> None:
    """Hydrating a persisted agent row that still carries a retired builtin now RAISES —
    we have migrated off (0122 + boot-gate residue proof), so there is no silent fallback."""
    from aios.db.queries.agents import _row_to_agent

    with pytest.raises(ValidationError):
        _row_to_agent(_agent_row([{"type": "bash"}, {"type": "complete_goal"}]))


def test_agent_row_with_current_vocabulary_hydrates() -> None:
    from aios.db.queries.agents import _row_to_agent

    agent = _row_to_agent(_agent_row([{"type": "bash"}, {"type": "read"}]))
    assert [t.type for t in agent.tools] == ["bash", "read"]
