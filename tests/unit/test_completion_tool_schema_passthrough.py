"""Regression coverage for provider-generic strict MCP tool forwarding."""

from __future__ import annotations

from aios.harness.completion import _build_litellm_kwargs


def test_tools_and_agent_extras_survive_completion_kwargs_unchanged() -> None:
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
                "strict": True,
            },
        }
    ]
    extra_body = {"provider": {"order": ["preferred", "fallback"]}}

    kwargs = _build_litellm_kwargs(
        model="openai-compatible/model",
        messages=[{"role": "user", "content": "Plan it"}],
        tools=tools,
        auth=None,
        extra={"reasoning_effort": "high", "extra_body": extra_body},
        session_id=None,
        stream=False,
    )

    assert kwargs["tools"] is tools
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == extra_body
    assert "response_format" not in kwargs
