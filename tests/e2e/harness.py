"""E2E test harness — real system, scripted model.

Provides the :class:`Harness` class (wired into tests via a pytest
fixture in ``conftest.py``) plus response builder helpers and assertion
helpers. A test author only needs to script the model's responses and
verify the event log.

Example::

    async def test_tool_round_trip(harness):
        harness.script_model([
            assistant(tool_calls=[bash("echo hello")]),
            assistant("The output was hello."),
        ])
        session = await harness.start("Say hello via bash")
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        assert "hello" in first_tool_result(events)["content"]
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import asyncpg

from aios.harness.context import should_call_model
from aios.harness.loop import run_session_step
from aios.harness.task_registry import TaskRegistry
from aios.models.events import Event
from aios.models.sessions import Session
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from aios.tools.registry import ToolHandler, registry

# ─── response builder helpers ────────────────────────────────────────────────


def assistant(
    content: str = "", *, tool_calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build a scripted assistant response dict."""
    d: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    return d


_call_counter = 0


def _next_call_id() -> str:
    global _call_counter
    _call_counter += 1
    return f"call_{_call_counter}"


def tool_call(
    name: str,
    arguments: dict[str, Any] | str = "{}",
    *,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build a tool_call dict for an assistant response."""
    args_str = json.dumps(arguments) if isinstance(arguments, dict) else arguments
    return {
        "id": call_id or _next_call_id(),
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def bash(command: str, *, call_id: str | None = None) -> dict[str, Any]:
    """Shorthand for tool_call("bash", {"command": ...})."""
    return tool_call("bash", {"command": command}, call_id=call_id)


def cancel(tool_call_id: str | None = None, *, call_id: str | None = None) -> dict[str, Any]:
    """Shorthand for tool_call("cancel", ...)."""
    args = {"tool_call_id": tool_call_id} if tool_call_id else {}
    return tool_call("cancel", args, call_id=call_id)


# ─── fake litellm response ──────────────────────────────────────────────────


class _FakeMessage:
    """Mimics the litellm Message object with model_dump()."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


# ─── assertion helpers ───────────────────────────────────────────────────────


def first_tool_result(events: list[Event]) -> dict[str, Any]:
    """Return the data dict of the first tool-role message event."""
    for e in events:
        if e.kind == "message" and e.data.get("role") == "tool":
            return e.data
    raise AssertionError("no tool result found in events")


def tool_results(events: list[Event]) -> list[dict[str, Any]]:
    """Return all tool-role message event data dicts."""
    return [e.data for e in events if e.kind == "message" and e.data.get("role") == "tool"]


def last_assistant_content(events: list[Event]) -> str:
    """Return the content of the last assistant message event."""
    for e in reversed(events):
        if e.kind == "message" and e.data.get("role") == "assistant":
            return e.data.get("content") or ""
    raise AssertionError("no assistant message found in events")


# ─── the harness ─────────────────────────────────────────────────────────────


class Harness:
    """E2E test harness. Real Postgres, real step function, scripted model."""

    def __init__(
        self,
        pool: asyncpg.Pool[Any],
        task_registry: TaskRegistry,
    ) -> None:
        self._pool = pool
        self._task_registry = task_registry
        self._responses: list[dict[str, Any]] = []
        self._response_idx = 0
        self._env_id: str | None = None
        self.model_calls: list[dict[str, Any]] = []  # captured kwargs from each litellm call

    # ── scripting ────────────────────────────────────────────────────────

    def script_model(self, responses: list[dict[str, Any]]) -> None:
        """Set the sequence of canned model responses."""
        global _call_counter
        _call_counter = 0
        self._responses = list(responses)
        self._response_idx = 0
        self.model_calls = []

    def register_tool(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str = "test tool",
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a custom tool for this test."""
        registry.register(
            name=name,
            description=description,
            parameters_schema=schema
            or {"type": "object", "properties": {}, "additionalProperties": True},
            handler=handler,
        )

    # ── session lifecycle ────────────────────────────────────────────────

    async def start(
        self,
        message: str,
        *,
        tools: list[str] | None = None,
        system: str = "You are a test assistant.",
    ) -> Session:
        """Create an agent + session and append the initial user message."""
        if self._env_id is None:
            from aios.ids import make_id

            env = await environments_service.create_environment(
                self._pool, name=f"test-env-{make_id('env')[-8:]}"
            )
            self._env_id = env.id

        from aios.models.agents import ToolSpec

        tool_specs = [ToolSpec(type=t) for t in (tools or [])]
        agent = await agents_service.create_agent(
            self._pool,
            name=f"test-agent-{make_id('agent')[-8:]}",
            model="fake/test",
            system=system,
            tools=tool_specs,
            credential_id=None,
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        session = await sessions_service.create_session(
            self._pool,
            agent_id=agent.id,
            environment_id=self._env_id,
            title="e2e-test",
            metadata={},
        )
        await sessions_service.append_user_message(self._pool, session.id, message)
        return await sessions_service.get_session(self._pool, session.id)

    async def inject_message(self, session_id: str, content: str) -> None:
        """Append a user message mid-turn."""
        await sessions_service.append_user_message(self._pool, session_id, content)

    # ── step execution ───────────────────────────────────────────────────

    async def run_step(self, session_id: str) -> None:
        """Run one inference step. Mocks are installed at fixture scope."""
        await run_session_step(session_id)

    async def wait_for_tools(self, session_id: str) -> None:
        """Wait for all in-flight tool tasks to complete."""
        for _ in range(100):  # safety cap
            tasks = self._task_registry._tasks.get(session_id, {})
            pending = [t for t in tasks.values() if not t.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)
        raise RuntimeError("wait_for_tools: tasks never completed")

    async def run_until_idle(self, session_id: str, *, max_steps: int = 20) -> None:
        """Run steps until should_call_model returns False."""
        for _ in range(max_steps):
            await self.wait_for_tools(session_id)
            events = await sessions_service.read_message_events(self._pool, session_id)
            if not should_call_model(events):
                return
            await self.run_step(session_id)
        raise RuntimeError(f"run_until_idle: hit max_steps={max_steps}")

    # ── inspection ───────────────────────────────────────────────────────

    async def events(self, session_id: str) -> list[Event]:
        """Read all message events for the session."""
        return await sessions_service.read_message_events(self._pool, session_id)

    async def session(self, session_id: str) -> Session:
        """Fetch the current session record."""
        return await sessions_service.get_session(self._pool, session_id)

    # ── internal ─────────────────────────────────────────────────────────

    def _pop_response(self, **kwargs: Any) -> dict[str, Any]:
        """Return the next scripted response wrapped in litellm envelope.

        Also records the kwargs (including ``messages``) on
        ``self.model_calls`` for test assertions about context shape.
        """
        self.model_calls.append(kwargs)
        if self._response_idx >= len(self._responses):
            raise AssertionError(
                f"model called {self._response_idx + 1} times but only "
                f"{len(self._responses)} responses were scripted"
            )
        resp = self._responses[self._response_idx]
        self._response_idx += 1
        return {"choices": [{"message": _FakeMessage(resp)}]}
