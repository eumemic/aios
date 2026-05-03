"""E2E test for the connector subprocess supervisor.

Spawns the in-tree :mod:`tests.fixtures.echo_connector` as a real
subprocess via :class:`~aios.harness.connector_supervisor.ConnectorSubprocessRegistry`
and exercises the four behaviours PR2 promises:

* ``initialize()`` succeeds and ``experimental.aios/connector`` validation passes.
* ``list_tools()`` round-trips through the live session.
* ``dispatch_call`` delivers a tool result.
* ``notifications/aios/accounts`` lands as an in-memory snapshot.
* The subprocess crashing triggers an automatic respawn.

No Postgres / Docker / harness fixtures — the supervisor's only
external dep is the subprocess, and the test owns the asyncio loop.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from aios.config import Settings
from aios.harness import connector_supervisor as supervisor_mod
from aios.harness.connector_supervisor import ConnectorSubprocessRegistry
from aios.mcp.stdio_transport import ConnectorSpec


def _echo_spec() -> ConnectorSpec:
    """Spawn ``python -m tests.fixtures.echo_connector`` from the current interpreter.

    Inheriting cwd from pytest puts the fixture module on ``sys.path``
    automatically.  Tracking the interpreter via ``sys.executable``
    keeps virtualenv selection consistent across CI variants.
    """
    return ConnectorSpec(
        name="echo",
        command=sys.executable,
        args=["-m", "tests.fixtures.echo_connector"],
    )


async def _wait_for(predicate: Any, *, max_wait_s: float = 10.0) -> None:
    """Poll ``predicate()`` every 50 ms until it returns truthy or we time out.

    Used in place of bespoke event/Condition wiring — the supervisor
    surfaces state changes as ``ConnectorState`` field updates, and
    polling is fine for a small handful of test asserts.
    """
    deadline = asyncio.get_event_loop().time() + max_wait_s
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"predicate {predicate!r} did not become true within {max_wait_s}s")


class TestConnectorSupervisor:
    async def test_starts_initializes_and_emits_accounts(self) -> None:
        """Happy path: spawn, init, capability check, account snapshot, shutdown."""
        registry = ConnectorSubprocessRegistry([_echo_spec()], settings=Settings())
        await registry.start()
        try:
            session = await asyncio.wait_for(registry.get_session("echo"), timeout=10.0)
            tools_result = await session.list_tools()
            tool_names = {t.name for t in tools_result.tools}
            assert tool_names == {"ping", "echo", "crash"}

            state = registry.state("echo")
            assert state is not None
            assert state.status == "running"
            assert state.instructions is not None
            assert "ping" in state.instructions

            await _wait_for(lambda: bool(state.accounts), max_wait_s=5.0)
            assert state.accounts == [{"id": "echo-1", "display_name": "Echo Account One"}]
        finally:
            await registry.shutdown()

    async def test_dispatch_call_round_trip(self) -> None:
        """``dispatch_call`` returns the connector's tool result envelope."""
        registry = ConnectorSubprocessRegistry([_echo_spec()], settings=Settings())
        await registry.start()
        try:
            await asyncio.wait_for(registry.get_session("echo"), timeout=10.0)
            ping = await registry.dispatch_call("echo", "ping", {})
            assert ping == {"content": "pong"}
            echoed = await registry.dispatch_call("echo", "echo", {"text": "hello"})
            assert echoed == {"content": "hello"}
        finally:
            await registry.shutdown()

    async def test_unknown_connector_returns_error_envelope(self) -> None:
        """Calls against an un-enabled connector surface as a coded envelope."""
        registry = ConnectorSubprocessRegistry([_echo_spec()], settings=Settings())
        await registry.start()
        try:
            await asyncio.wait_for(registry.get_session("echo"), timeout=10.0)
            result = await registry.dispatch_call("nonexistent", "ping", {})
            assert "not enabled" in result["error"]
            assert result["code"] == "not_enabled"
        finally:
            await registry.shutdown()

    async def test_subprocess_crash_triggers_respawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A connector that exits unexpectedly is restarted by the supervisor.

        Patches the initial backoff to 0.5 s so the restart lands inside
        the test's ceiling.  We capture the original session by id and
        wait for state.session to *change* — checking ``is not None``
        alone races: the supervisor task hasn't yet observed the
        subprocess EOF when ``dispatch_call`` returns, so the stale
        session reference is still in place for a beat.
        """
        monkeypatch.setattr(supervisor_mod, "_BACKOFF_INITIAL_S", 0.5)
        registry = ConnectorSubprocessRegistry([_echo_spec()], settings=Settings())
        await registry.start()
        try:
            await asyncio.wait_for(registry.get_session("echo"), timeout=10.0)
            state = registry.state("echo")
            assert state is not None
            await _wait_for(lambda: bool(state.accounts), max_wait_s=5.0)
            original_session = state.session

            # Trigger the fixture to ``os._exit(1)``.  The dispatch_call
            # surfaces the closed transport as a transport error envelope.
            crash_result = await registry.dispatch_call("echo", "crash", {})
            assert "error" in crash_result
            assert crash_result["code"] == "transport_error"

            # Wait for the supervisor to spawn a *different* session and
            # flip back to ``running``.
            await _wait_for(
                lambda: (
                    state.status == "running"
                    and state.session is not None
                    and state.session is not original_session
                ),
                max_wait_s=15.0,
            )
            # Fresh subprocess should still respond.
            ping = await registry.dispatch_call("echo", "ping", {})
            assert ping == {"content": "pong"}
        finally:
            await registry.shutdown()
