"""The per-step prelude's provider-tool clamp branch (#1627).

Unit-level: no Postgres, no Docker. ``compute_step_prelude`` only touches the pool for
``get_open_obligations`` (one acquire), so we hand it a stub pool whose acquired
connection returns no obligations, a stub ``ToolProvider`` that injects one known custom
tool, and an in-memory ``Agent``. The two cases nail the new branch:

* foreground session (``parent_run_id is None``) → the provider tool passes through
  UNCHANGED (today's connector UX; no regression);
* born-clamped child (``parent_run_id`` set) whose ``agent.tools`` (the frozen effective
  surface) excludes that tool's ``_tool_key`` → the tool is DROPPED. This is the headline
  assertion that the #794-class ambient-authority hole is closed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.harness.step_context import compute_step_prelude
from aios.models.agents import Agent, ToolSpec

pytestmark = pytest.mark.asyncio

_ACCOUNT = "acc_1627"
_SESSION = "sess_1627"

# The connector-injected custom tool the ToolProvider returns each step.
_PROVIDER_TOOL = {
    "type": "custom",
    "name": "connector_send",
    "description": "send via the connector",
    "input_schema": {"type": "object"},
}


def _agent(tools: list[ToolSpec]) -> Agent:
    now = datetime.now(UTC)
    return Agent(
        id="agt_1627",
        version=1,
        name="a",
        model="gpt-test",
        system="you are a test agent",
        tools=tools,
        mcp_servers=[],
        http_servers=[],
        description=None,
        metadata={},
        window_min=1,
        window_max=10,
        created_at=now,
        updated_at=now,
    )


def _session(*, parent_run_id: str | None) -> Any:
    """A minimal stand-in carrying the only field the clamp branch reads."""
    return mock.Mock(id=_SESSION, parent_run_id=parent_run_id)


class _StubConn:
    async def __aenter__(self) -> _StubConn:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _StubPool:
    """Only ``acquire()`` is exercised (for ``get_open_obligations``)."""

    def acquire(self) -> _StubConn:
        return _StubConn()


def _stub_tool_provider() -> Any:
    tp = mock.Mock()
    tp.list_tools_for_session = AsyncMock(return_value=[dict(_PROVIDER_TOOL)])
    return tp


async def _prelude_tool_names(agent: Agent, session: Any) -> list[str]:
    prev = runtime.tool_provider
    runtime.tool_provider = _stub_tool_provider()
    try:
        with mock.patch(
            "aios.db.queries.get_open_obligations", new=AsyncMock(return_value=[])
        ):
            prelude = await compute_step_prelude(
                _StubPool(),  # type: ignore[arg-type]
                _SESSION,
                account_id=_ACCOUNT,
                session=session,
                agent=agent,
                channels=[],
                memory_store_echoes=[],
            )
    finally:
        runtime.tool_provider = prev
    return [t["function"]["name"] for t in prelude.tools]


async def test_prelude_foreground_admits_provider_tool() -> None:
    """A foreground session (``parent_run_id is None``) keeps the injected provider tool."""
    # The agent did NOT declare the connector tool (foreground declares its own surface).
    agent = _agent(tools=[ToolSpec(type="bash")])
    names = await _prelude_tool_names(agent, _session(parent_run_id=None))
    assert "connector_send" in names


async def test_prelude_clamped_child_drops_provider_tool() -> None:
    """HEADLINE (#1627): a born-clamped child whose frozen surface excludes the provider
    tool's ``_tool_key`` does NOT receive it — the ToolProvider seam can't re-grant it."""
    # agent.tools is the overlaid frozen effective surface; it excludes connector_send.
    agent = _agent(tools=[ToolSpec(type="bash")])
    names = await _prelude_tool_names(agent, _session(parent_run_id="run_abc"))
    assert "connector_send" not in names


async def test_prelude_clamped_child_admits_declared_provider_tool() -> None:
    """A clamped child whose frozen surface DOES carry the connector tool's key keeps it
    (the run granted it) — proving the gate is a clamp, not a blanket drop."""
    agent = _agent(
        tools=[
            ToolSpec(type="bash"),
            ToolSpec(
                type="custom",
                name="connector_send",
                description="send via the connector",
                input_schema={"type": "object"},
            ),
        ]
    )
    names = await _prelude_tool_names(agent, _session(parent_run_id="run_abc"))
    assert "connector_send" in names
