"""Regression coverage for provider-generic strict MCP tool forwarding."""

from __future__ import annotations

import pytest

from aios.harness.completion import _build_litellm_kwargs


@pytest.mark.parametrize(
    "model",
    ["openai/test-model", "anthropic/test-model", "xai/test-model"],
)
def test_tools_and_agent_extras_survive_provider_routes_unchanged(model: str) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "mcp__planner__propose",
                "description": "Propose a plan",
                "parameters": {
                    "type": "object",
                    "properties": {"draft": {"type": "object"}},
                    "required": ["draft"],
                },
                "strict": False,
            },
        }
    ]
    extra_body = {"provider": {"order": ["preferred", "fallback"]}}

    kwargs = _build_litellm_kwargs(
        model=model,
        messages=[{"role": "user", "content": "Plan it"}],
        tools=tools,
        auth=None,
        extra={"reasoning_effort": "high", "extra_body": extra_body},
        session_id=None,
        stream=False,
    )

    assert kwargs["tools"] is tools
    forwarded_tools = kwargs["tools"]
    assert isinstance(forwarded_tools, list)
    forwarded_function = forwarded_tools[0]["function"]
    assert isinstance(forwarded_function, dict)
    assert forwarded_function["strict"] is False
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == extra_body
    assert "response_format" not in kwargs
