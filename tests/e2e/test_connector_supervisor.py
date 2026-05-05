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

import pytest
from aios_connector import ConnectorSpec

from aios.config import ConnectorInstance, Settings
from aios.harness import connector_supervisor as supervisor_mod
from aios.harness.connector_supervisor import ConnectorSubprocessRegistry
from tests.e2e.conftest import wait_for_predicate as _wait_for


def _echo_specs() -> list[tuple[ConnectorInstance, ConnectorSpec]]:
    """Spawn ``python -m tests.fixtures.echo_connector`` as the default ``echo`` instance.

    Inheriting cwd from pytest puts the fixture module on ``sys.path``
    automatically.  Tracking the interpreter via ``sys.executable``
    keeps virtualenv selection consistent across CI variants.

    Default-instance shape (``instance == connector``) so the tests
    exercise the same code path as ``connectors_enabled=echo``
    single-instance setups.
    """
    return [
        (
            ConnectorInstance(connector="echo", instance="echo"),
            ConnectorSpec(
                name="echo",
                command=sys.executable,
                args=["-m", "tests.fixtures.echo_connector"],
            ),
        )
    ]


def _multi_instance_echo_specs() -> list[tuple[ConnectorInstance, ConnectorSpec]]:
    """Two echo instances, one connector type — exercises multi-instance keying."""
    base_args = ["-m", "tests.fixtures.echo_connector"]
    return [
        (
            ConnectorInstance(connector="echo", instance="alpha"),
            ConnectorSpec(name="echo", command=sys.executable, args=base_args),
        ),
        (
            ConnectorInstance(connector="echo", instance="beta"),
            ConnectorSpec(name="echo", command=sys.executable, args=base_args),
        ),
    ]


class TestConnectorSupervisor:
    async def test_starts_initializes_and_emits_accounts(self) -> None:
        """Happy path: spawn, init, capability check, account snapshot, shutdown."""
        registry = ConnectorSubprocessRegistry(_echo_specs(), settings=Settings())
        await registry.start()
        try:
            session = await asyncio.wait_for(registry.get_session("echo", "echo"), timeout=10.0)
            tools_result = await session.list_tools()
            tool_names = {t.name for t in tools_result.tools}
            assert tool_names == {"ping", "echo", "crash"}

            state = registry.state("echo", "echo")
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
        registry = ConnectorSubprocessRegistry(_echo_specs(), settings=Settings())
        await registry.start()
        try:
            await asyncio.wait_for(registry.get_session("echo", "echo"), timeout=10.0)
            ping = await registry.dispatch_call("echo", "echo", "ping", {})
            assert ping == {"content": "pong"}
            echoed = await registry.dispatch_call("echo", "echo", "echo", {"text": "hello"})
            assert echoed == {"content": "hello"}
        finally:
            await registry.shutdown()

    async def test_unknown_connector_returns_error_envelope(self) -> None:
        """Calls against an un-enabled instance surface as a coded envelope."""
        registry = ConnectorSubprocessRegistry(_echo_specs(), settings=Settings())
        await registry.start()
        try:
            await asyncio.wait_for(registry.get_session("echo", "echo"), timeout=10.0)
            result = await registry.dispatch_call("nonexistent", "x", "ping", {})
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
        registry = ConnectorSubprocessRegistry(_echo_specs(), settings=Settings())
        await registry.start()
        try:
            await asyncio.wait_for(registry.get_session("echo", "echo"), timeout=10.0)
            state = registry.state("echo", "echo")
            assert state is not None
            await _wait_for(lambda: bool(state.accounts), max_wait_s=5.0)
            original_session = state.session

            # Trigger the fixture to ``os._exit(1)``.  The dispatch_call
            # surfaces the closed transport as a transport error envelope.
            crash_result = await registry.dispatch_call("echo", "echo", "crash", {})
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
            ping = await registry.dispatch_call("echo", "echo", "ping", {})
            assert ping == {"content": "pong"}
        finally:
            await registry.shutdown()


class TestMultiInstanceSupervisor:
    """Two instances of the same connector type spawn as independent subprocesses.

    Verifies the registry's compound keying actually produces N distinct
    supervisor tasks + sessions, and that one instance's failure doesn't
    propagate to the sibling.
    """

    async def test_two_instances_spawn_with_distinct_sessions(self) -> None:
        registry = ConnectorSubprocessRegistry(_multi_instance_echo_specs(), settings=Settings())
        await registry.start()
        try:
            session_alpha = await asyncio.wait_for(
                registry.get_session("echo", "alpha"), timeout=10.0
            )
            session_beta = await asyncio.wait_for(
                registry.get_session("echo", "beta"), timeout=10.0
            )
            # Distinct OS subprocesses means distinct ClientSession objects.
            assert session_alpha is not session_beta

            # Both report as ``running`` and snapshot lists both.
            states = registry.snapshot_all()
            statuses = {(s["connector"], s["instance"]): s["status"] for s in states}
            assert statuses == {("echo", "alpha"): "running", ("echo", "beta"): "running"}

            # Each subprocess responds to ping independently.
            ping_a = await registry.dispatch_call("echo", "alpha", "ping", {})
            ping_b = await registry.dispatch_call("echo", "beta", "ping", {})
            assert ping_a == {"content": "pong"}
            assert ping_b == {"content": "pong"}
        finally:
            await registry.shutdown()

    async def test_one_instance_crash_does_not_take_down_sibling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(supervisor_mod, "_BACKOFF_INITIAL_S", 0.5)
        registry = ConnectorSubprocessRegistry(_multi_instance_echo_specs(), settings=Settings())
        await registry.start()
        try:
            await asyncio.wait_for(registry.get_session("echo", "alpha"), timeout=10.0)
            await asyncio.wait_for(registry.get_session("echo", "beta"), timeout=10.0)
            beta_state = registry.state("echo", "beta")
            assert beta_state is not None
            beta_session_before = beta_state.session

            # Crash alpha — beta should remain ``running`` throughout.
            crash_result = await registry.dispatch_call("echo", "alpha", "crash", {})
            assert crash_result["code"] == "transport_error"

            # Beta's session reference must NOT change (no respawn for siblings).
            await asyncio.sleep(0.5)
            assert beta_state.status == "running"
            assert beta_state.session is beta_session_before

            # Beta still responds to ping while alpha is restarting.
            ping_b = await registry.dispatch_call("echo", "beta", "ping", {})
            assert ping_b == {"content": "pong"}
        finally:
            await registry.shutdown()
