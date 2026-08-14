"""Fail-closed hydration for goal-outcome builtins retired by migration 0122."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.db.queries.agents import _row_to_agent


@pytest.mark.parametrize("retired", ["complete_goal", "fail_goal"])
def test_agent_row_with_retired_builtin_fails_hydration(retired: str) -> None:
    """Migration 0122 closed the compatibility window; stale persisted tools fail closed."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    row = {
        "id": "agt_stale",
        "version": 3,
        "name": "stale",
        "model": "anthropic/claude-opus-4-6",
        "system": "",
        "tools": [{"type": "bash"}, {"type": retired}],
        "skills": [],
        "mcp_servers": [],
        "http_servers": [],
        "description": None,
        "metadata": {},
        "litellm_extra": {},
        "window_min": 1,
        "window_max": 10,
        "preempt_policy": "wait",
        "created_by_type": None,
        "created_by_ref": None,
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
    }

    with pytest.raises(ValidationError):
        _row_to_agent(row)
