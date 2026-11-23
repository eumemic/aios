from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import litellm
from fastapi import Response

from aios.api.routers import agents as router
from aios.models.agents import Agent, AgentUpdate
from aios.services import agents as agents_service


def _agent(**over: Any) -> Agent:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    values: dict[str, Any] = {
        "id": "agt_1",
        "version": 1,
        "name": "agent",
        "model": "xai/grok-4.6",
        "system": "",
        "tools": [],
        "skills": [],
        "mcp_servers": [],
        "http_servers": [],
        "description": None,
        "metadata": {},
        "litellm_extra": {},
        "window_min": 1,
        "window_max": 2,
        "created_at": now,
        "updated_at": now,
    }
    values.update(over)
    return Agent(**values)


async def test_update_warns_when_local_map_would_reject_param(monkeypatch: Any) -> None:
    # Model metadata may refresh independently of the locked LiteLLM package. Pin the
    # stale-map condition this warning is specifically intended to report.
    monkeypatch.setattr(
        litellm,
        "get_supported_openai_params",
        lambda _model: ["temperature"],
    )
    current = _agent()
    updated = _agent(version=2, litellm_extra={"reasoning_effort": "high"})
    monkeypatch.setattr(agents_service, "get_agent", AsyncMock(return_value=current))
    monkeypatch.setattr(agents_service, "update_agent", AsyncMock(return_value=updated))
    response = Response()

    result = await router.update(
        "agt_1",
        AgentUpdate(version=1, litellm_extra={"reasoning_effort": "high"}),
        object(),
        "acc_1",
        response,
    )

    assert result == updated
    assert "reasoning_effort" in response.headers["Warning"]
    assert "pass them to the provider" in response.headers["Warning"]


async def test_update_normalization_is_visible_in_response(monkeypatch: Any) -> None:
    # #2013: the stored model differs from what the caller typed — the response
    # must say so rather than mutate silently.
    current = _agent()
    updated = _agent(version=2, model="openai/responses/gpt-5.6-sol")
    monkeypatch.setattr(agents_service, "get_agent", AsyncMock(return_value=current))
    monkeypatch.setattr(agents_service, "update_agent", AsyncMock(return_value=updated))
    response = Response()

    result = await router.update(
        "agt_1", AgentUpdate(version=1, model="gpt-5.6-sol"), object(), "acc_1", response
    )

    assert result.model == "openai/responses/gpt-5.6-sol"
    assert response.headers[router.MODEL_ROUTE_HEADER] == (
        "gpt-5.6-sol -> openai/responses/gpt-5.6-sol"
    )
    assert "normalized to canonical route" in response.headers["Warning"]


async def test_update_without_normalization_sets_no_route_header(monkeypatch: Any) -> None:
    current = _agent()
    updated = _agent(version=2, model="openai/gpt-4o")
    monkeypatch.setattr(agents_service, "get_agent", AsyncMock(return_value=current))
    monkeypatch.setattr(agents_service, "update_agent", AsyncMock(return_value=updated))
    response = Response()

    await router.update(
        "agt_1", AgentUpdate(version=1, model="openai/gpt-4o"), object(), "acc_1", response
    )

    assert router.MODEL_ROUTE_HEADER not in response.headers
