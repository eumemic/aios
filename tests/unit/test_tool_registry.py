"""Unit tests for the tool registry + OpenAI schema translator.

Pure in-memory — no Postgres, no Docker. Uses a fresh registry per test
via ``registry.clear()`` so tests don't poison each other.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.models.agents import ToolSpec as AgentToolSpec
from aios.tools.registry import (
    DuplicateToolError,
    ToolNotFoundError,
    registry,
    to_openai_tools,
)


async def _noop_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Snapshot the registry before each test and restore after.

    The real aios package imports ``aios.tools`` at worker startup
    which registers the built-in bash tool. Some tests need a clean
    slate; others want to assert against the registered built-ins.
    We snapshot and restore to keep both paths happy.
    """
    snapshot = dict(registry._tools)
    try:
        yield
    finally:
        registry._tools = snapshot


class TestRegister:
    def test_register_stores_definition(self) -> None:
        registry.clear()
        registry.register(
            name="dummy",
            description="A dummy tool.",
            parameters_schema={"type": "object"},
            handler=_noop_handler,
        )
        assert registry.has("dummy")
        defn = registry.get("dummy")
        assert defn.name == "dummy"
        assert defn.description == "A dummy tool."

    def test_register_rejects_duplicate_name(self) -> None:
        registry.clear()
        registry.register(
            name="dup",
            description="first",
            parameters_schema={"type": "object"},
            handler=_noop_handler,
        )
        with pytest.raises(DuplicateToolError):
            registry.register(
                name="dup",
                description="second",
                parameters_schema={"type": "object"},
                handler=_noop_handler,
            )

    def test_get_unknown_raises(self) -> None:
        registry.clear()
        with pytest.raises(ToolNotFoundError):
            registry.get("nonexistent")

    def test_names_returns_sorted(self) -> None:
        registry.clear()
        for name in ["zebra", "apple", "mango"]:
            registry.register(
                name=name,
                description=name,
                parameters_schema={"type": "object"},
                handler=_noop_handler,
            )
        assert registry.names() == ["apple", "mango", "zebra"]


class TestBashToolRegistered:
    """The bash tool should be registered after importing aios.tools."""

    def test_bash_is_registered(self) -> None:
        # Trigger the side-effect import if it hasn't run yet.
        import aios.tools  # noqa: F401

        assert registry.has("bash")
        defn = registry.get("bash")
        assert defn.name == "bash"
        # Parameters schema shape sanity.
        params = defn.parameters_schema
        assert params["type"] == "object"
        assert "command" in params["properties"]
        assert params["required"] == ["command"]


class TestToOpenaiTools:
    def test_empty_tools_returns_empty_list(self) -> None:
        result = to_openai_tools([])
        assert result == []

    def test_single_tool_translated(self) -> None:
        # Make sure bash is registered (via side-effect import).
        import aios.tools  # noqa: F401

        result = to_openai_tools([AgentToolSpec(type="bash")])
        assert len(result) == 1
        entry = result[0]
        assert entry["type"] == "function"
        assert entry["function"]["name"] == "bash"
        assert "description" in entry["function"]
        assert entry["function"]["parameters"]["type"] == "object"
        assert "command" in entry["function"]["parameters"]["properties"]

    def test_unknown_tool_raises(self) -> None:
        registry.clear()
        # Register a tool that isn't 'bash' so the bash entry is absent.
        registry.register(
            name="something",
            description="x",
            parameters_schema={"type": "object"},
            handler=_noop_handler,
        )
        with pytest.raises(ToolNotFoundError):
            to_openai_tools([AgentToolSpec(type="bash")])
