"""Tool registry — name → ToolSpec (JSON schema + handler).

Design lifted from ``~/code/hermes-agent/tools/registry.py``: a module-level
singleton that tool modules register themselves against at import time.
The harness imports :mod:`aios.tools` once at worker startup, which causes
every built-in tool module to run its registration call.

Each :class:`ToolSpec` holds:

* ``name`` — the string the model uses in ``tool_calls[*].function.name``
* ``description`` — what the tool does, shown to the model
* ``parameters_schema`` — JSON Schema describing the ``arguments`` shape
* ``handler`` — an async callable that runs the tool

The :func:`to_openai_tools` helper translates an agent's ``tools`` list
(``[{type: 'bash'}, {type: 'read'}, ...]``) into the
chat-completions/OpenAI ``tools`` parameter that LiteLLM expects.
Translation is lookup + shape conversion — no per-tool customization yet.
Later phases may add per-tool config (``{type: 'bash', timeout: 60}``)
but the registry shape won't change.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aios.errors import AiosError
from aios.models.agents import ToolSpec as AgentToolSpec


class ToolNotFoundError(AiosError):
    """Raised when a tool name is referenced but not registered."""

    error_type = "tool_not_found"
    status_code = 404


class DuplicateToolError(AiosError):
    """Raised when a tool tries to register under a name already in use."""

    error_type = "duplicate_tool"
    status_code = 500


# Handler signature: async (session_id, arguments) -> result dict.
# The result dict is appended verbatim as the ``content`` or ``error`` of
# a tool-role message. Handlers do NOT call append_event themselves —
# :mod:`aios.harness.tool_dispatch` does that.
ToolHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    """A registered tool's full definition."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: ToolHandler


@dataclass(slots=True)
class ToolRegistry:
    """Singleton registry. Import :data:`registry` from this module."""

    _tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        """Register a tool. Raises :class:`DuplicateToolError` on name clash.

        Intended to be called at module import time from tool modules
        (e.g. :mod:`aios.tools.bash`).
        """
        if name in self._tools:
            raise DuplicateToolError(
                f"tool {name!r} already registered",
                detail={"name": name},
            )
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            handler=handler,
        )

    def get(self, name: str) -> ToolDefinition:
        """Return the registered tool. Raises :class:`ToolNotFoundError` if absent."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"tool {name!r} is not registered",
                detail={"name": name},
            )
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def clear(self) -> None:
        """Drop all registrations. Used by unit tests."""
        self._tools.clear()


registry = ToolRegistry()


def to_openai_tools(agent_tools: list[AgentToolSpec]) -> list[dict[str, Any]]:
    """Translate an agent's declared tools into the OpenAI ``tools`` param.

    Takes the agent's ``tools: [{type: 'bash'}, {type: 'read'}, ...]`` list
    and returns the chat-completions shape:

    .. code-block::

        [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "...",
                    "parameters": {<json schema>},
                },
            },
            ...
        ]

    Unknown tool types raise :class:`ToolNotFoundError`. If the agent
    declares no tools, returns an empty list.
    """
    result: list[dict[str, Any]] = []
    for entry in agent_tools:
        if entry.type == "custom":
            # Custom tools carry their own schema — not in the registry.
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": entry.name,
                        "description": entry.description or "",
                        "parameters": entry.input_schema or {},
                    },
                }
            )
        else:
            tool = registry.get(entry.type)
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_schema,
                    },
                }
            )
    return result
