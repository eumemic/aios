"""Read tolerance for MCP server URLs persisted before current write rules."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aios.db.queries.agents import _row_to_agent
from aios.models.agents import McpServerSpec


def test_persisted_loopback_mcp_server_hydrates_without_weakening_writes() -> None:
    now = datetime.now(UTC)
    legacy_server = {"type": "url", "name": "signal", "url": "http://localhost:8091/mcp"}
    row: dict[str, object] = {
        "id": "agt_legacy_mcp",
        "version": 1,
        "name": "legacy",
        "model": "openai/gpt-4o",
        "system": "",
        "tools": [],
        "skills": [],
        # Deliberately bypass current request validation, as an old JSONB row does.
        "mcp_servers": [legacy_server],
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

    agent = _row_to_agent(row)

    assert agent.mcp_servers[0].url == "http://localhost:8091/mcp"
    with pytest.raises(ValidationError, match="private, internal, or runtime-local"):
        McpServerSpec.model_validate(legacy_server)
