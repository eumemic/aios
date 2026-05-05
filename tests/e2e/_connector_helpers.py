"""Shared helpers for e2e tests that drive ``ConnectorSubprocessRegistry``
directly without spawning a real connector subprocess.
"""

from __future__ import annotations

from typing import Any

from aios.config import ConnectorInstance, Settings
from aios.harness.connector_supervisor import ConnectorSubprocessRegistry
from aios.ids import make_id
from aios.mcp.stdio_transport import ConnectorSpec
from aios.models.agents import ToolSpec
from aios.services import (
    agents as agents_service,
)
from aios.services import (
    environments as environments_service,
)
from tests.e2e.harness import Harness


def make_spec(name: str = "echo") -> ConnectorSpec:
    """A spec the supervisor can hold without spawning anything.

    Tests never call ``start()`` so command/cwd never run; we only
    need a name so :class:`ConnectorState` initializes cleanly.
    """
    return ConnectorSpec(name=name, command="x", args=[])


def make_registry(
    settings: Settings | None = None, name: str = "echo"
) -> ConnectorSubprocessRegistry:
    """Build a registry with one default-instance entry for tests."""
    settings = settings or Settings()
    return ConnectorSubprocessRegistry(
        [(ConnectorInstance(connector=name, instance=name), make_spec(name))],
        settings=settings,
    )


def unique_account(prefix: str = "acct") -> str:
    """Per-test account string so testcontainer state stays isolated.

    The active-connection unique index would otherwise reject the
    second test that re-uses the same account (the testcontainer
    Postgres persists across tests in a session).
    """
    return f"{prefix}-{make_id('evt')[-10:]}"


def patch_send_ack(registry: ConnectorSubprocessRegistry) -> list[str]:
    """Replace ``_send_ack`` with a recorder.

    The real implementation calls ``dispatch_call`` against the
    (absent) connector subprocess, which would error. Returns the
    list that records each acked event_id.
    """
    recorded: list[str] = []

    async def fake_ack(self: Any, state: Any, event_id: str) -> None:
        recorded.append(event_id)

    registry._send_ack = fake_ack.__get__(registry, type(registry))  # type: ignore[method-assign]
    return recorded


async def make_agent_and_env(harness: Harness, *, prefix: str = "test") -> tuple[str, str]:
    """Build a fake agent + environment, returning their ids."""
    if harness._env_id is None:
        env = await environments_service.create_environment(
            harness._pool, name=f"{prefix}-env-{make_id('env')[-8:]}"
        )
        harness._env_id = env.id
    agent = await agents_service.create_agent(
        harness._pool,
        name=f"{prefix}-agent-{make_id('agent')[-8:]}",
        model="fake/test",
        system="test",
        tools=[ToolSpec(type="bash")],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    return agent.id, harness._env_id
