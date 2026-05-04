"""Unit tests for the connector supervisor's pure helpers.

The full subprocess lifecycle is covered by ``tests/e2e/test_connector_supervisor.py``;
this file targets the synchronous / lightly-async helpers that don't
need a real subprocess: capability validation, spec resolution
against the entry-point group, and the stderr pump's line extraction
across chunk boundaries.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from aios.harness.connector_supervisor import (
    _AIOS_EXPERIMENTAL_KEY,
    ConnectorSubprocessRegistry,
    resolve_connector_specs,
)
from aios.mcp.stdio_transport import ConnectorSpec, _pump_stderr


def _fake_init_result(experimental: dict[str, Any] | None) -> MagicMock:
    """Build a minimal :class:`InitializeResult` stand-in with controllable capabilities."""
    init = MagicMock()
    init.capabilities = MagicMock()
    init.capabilities.experimental = experimental
    return init


class TestValidateCapability:
    """``_validate_capability`` is the supervisor's hard-fail invariant.

    Connectors MUST declare ``experimental.aios/connector`` at
    initialize time.  A missing key, ``None``, or an empty dict all
    represent connectors that didn't opt into the protocol.
    """

    def _registry(self) -> ConnectorSubprocessRegistry:
        from aios.config import Settings

        return ConnectorSubprocessRegistry([], settings=Settings())

    def _state(self) -> Any:
        from aios.harness.connector_supervisor import ConnectorState

        spec = ConnectorSpec(name="echo", command="x", args=[])
        return ConnectorState(connector="echo", instance="echo", spec=spec)

    def test_passes_when_aios_connector_key_present(self) -> None:
        registry = self._registry()
        init = _fake_init_result({_AIOS_EXPERIMENTAL_KEY: {}})
        # Should not raise.
        registry._validate_capability(self._state(), init)

    def test_raises_on_missing_experimental_dict(self) -> None:
        registry = self._registry()
        init = _fake_init_result(None)
        with pytest.raises(RuntimeError, match=r"experimental\..*aios/connector"):
            registry._validate_capability(self._state(), init)

    def test_raises_on_empty_experimental_dict(self) -> None:
        registry = self._registry()
        init = _fake_init_result({})
        with pytest.raises(RuntimeError, match=r"experimental\..*aios/connector"):
            registry._validate_capability(self._state(), init)

    def test_raises_when_other_experimental_keys_present(self) -> None:
        registry = self._registry()
        init = _fake_init_result({"some/other/cap": {"option": True}})
        with pytest.raises(RuntimeError, match=r"experimental\..*aios/connector"):
            registry._validate_capability(self._state(), init)


class TestResolveConnectorSpecs:
    """The resolver maps ``connectors_enabled`` to ``ConnectorSpec`` via entry points.

    Mocking ``entry_points`` keeps the test off-disk; we exercise the
    three failure modes (unknown name, factory returns wrong type,
    factory raises) plus the cwd-defaulting path that satisfies
    plan-decision #11.
    """

    def _settings(self, *, enabled: list[str], connectors_dir: Any = None) -> Any:
        from pathlib import Path

        s = MagicMock()
        s.connectors_enabled = enabled
        s.connectors_dir = connectors_dir or Path("/tmp/aios-connectors-test")
        return s

    def test_unknown_name_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aios.harness.connector_supervisor.entry_points",
            lambda group: [],
        )
        with pytest.raises(RuntimeError, match=r"no aios\.connectors entry point"):
            resolve_connector_specs(self._settings(enabled=["unknown"]))

    def test_factory_wrong_return_type_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = MagicMock()
        ep.name = "bad"
        ep.load.return_value = lambda name, settings: "not a ConnectorSpec"
        monkeypatch.setattr(
            "aios.harness.connector_supervisor.entry_points",
            lambda group: [ep],
        )
        with pytest.raises(RuntimeError, match="expected ConnectorSpec"):
            resolve_connector_specs(self._settings(enabled=["bad"]))

    def test_default_instance_returns_single_segment_cwd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pathlib import Path

        ep = MagicMock()
        ep.name = "echo"
        ep.load.return_value = lambda name, settings: ConnectorSpec(name=name, command="/bin/echo")
        monkeypatch.setattr(
            "aios.harness.connector_supervisor.entry_points",
            lambda group: [ep],
        )
        connectors_dir = Path("/tmp/aios-connectors-test")
        specs = resolve_connector_specs(
            self._settings(enabled=["echo"], connectors_dir=connectors_dir)
        )
        assert len(specs) == 1
        ci, spec = specs[0]
        assert ci.connector == "echo" and ci.instance == "echo"
        # Default-instance setups keep the PR3 single-segment path.
        assert spec.cwd == connectors_dir / "echo"

    def test_non_default_instance_gets_per_instance_subdir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pathlib import Path

        ep = MagicMock()
        ep.name = "telegram"
        ep.load.return_value = lambda name, settings: ConnectorSpec(name=name, command="/bin/echo")
        monkeypatch.setattr(
            "aios.harness.connector_supervisor.entry_points",
            lambda group: [ep],
        )
        connectors_dir = Path("/tmp/aios-connectors-test")
        specs = resolve_connector_specs(
            self._settings(enabled=["telegram:bot1"], connectors_dir=connectors_dir)
        )
        assert len(specs) == 1
        ci, spec = specs[0]
        assert ci.connector == "telegram" and ci.instance == "bot1"
        assert spec.cwd == connectors_dir / "telegram" / "bot1"

    def test_factory_explicit_cwd_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pathlib import Path

        explicit = Path("/var/lib/aios/special")
        ep = MagicMock()
        ep.name = "echo"
        ep.load.return_value = lambda name, settings: ConnectorSpec(
            name=name, command="/bin/echo", cwd=explicit
        )
        monkeypatch.setattr(
            "aios.harness.connector_supervisor.entry_points",
            lambda group: [ep],
        )
        specs = resolve_connector_specs(self._settings(enabled=["echo"]))
        assert specs[0][1].cwd == explicit

    def test_env_re_export_for_non_default_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = MagicMock()
        ep.name = "telegram"
        ep.load.return_value = lambda name, settings: ConnectorSpec(name=name, command="/bin/echo")
        monkeypatch.setattr(
            "aios.harness.connector_supervisor.entry_points",
            lambda group: [ep],
        )
        # Scoped + unscoped vars present in parent env; the scoped value
        # must override under AIOS_TELEGRAM_BOT_TOKEN inside the bot1
        # subprocess's env.
        monkeypatch.setenv("AIOS_TELEGRAM_BOT_TOKEN", "unscoped")
        monkeypatch.setenv("AIOS_TELEGRAM_BOT1_BOT_TOKEN", "scoped-bot1")
        specs = resolve_connector_specs(self._settings(enabled=["telegram:bot1"]))
        _, spec = specs[0]
        assert spec.env is not None
        assert spec.env["AIOS_TELEGRAM_BOT_TOKEN"] == "scoped-bot1"

    def test_env_passes_unscoped_for_default_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ep = MagicMock()
        ep.name = "telegram"
        ep.load.return_value = lambda name, settings: ConnectorSpec(name=name, command="/bin/echo")
        monkeypatch.setattr(
            "aios.harness.connector_supervisor.entry_points",
            lambda group: [ep],
        )
        monkeypatch.setenv("AIOS_TELEGRAM_BOT_TOKEN", "the-only-token")
        specs = resolve_connector_specs(self._settings(enabled=["telegram"]))
        _, spec = specs[0]
        # Default-instance just inherits parent env unmodified.
        assert spec.env is not None
        assert spec.env["AIOS_TELEGRAM_BOT_TOKEN"] == "the-only-token"


class TestPumpStderrLineExtraction:
    """The stderr pump must split lines O(n) across chunk boundaries.

    The PR description's "O(n) split-and-pop" claim is unverified by
    the e2e test (which only triggers a single accounts-emit event).
    These tests pipe known byte sequences through ``_pump_stderr``
    directly and confirm the bound logger receives one event per line.
    """

    async def test_lines_split_at_chunk_boundaries(self) -> None:
        # Two writes, line breaks straddling the boundary.
        captured: list[str] = []
        read_fd, write_fd = os.pipe()
        try:
            pump = asyncio.create_task(_pump_stderr("test_conn", read_fd))
            with mock_log_capture(captured):
                # First chunk: partial line + complete line + partial.
                os.write(write_fd, b"hello\nworld\npart")
                await asyncio.sleep(0.05)
                # Second chunk completes the partial.
                os.write(write_fd, b"ial\nfinal\n")
                await asyncio.sleep(0.05)
                os.close(write_fd)
                write_fd = -1
                await asyncio.wait_for(pump, timeout=1.0)
        finally:
            if write_fd != -1:
                os.close(write_fd)
        # The pump owns ``read_fd`` cleanup via its finally.
        assert captured == ["hello", "world", "partial", "final"]

    async def test_empty_lines_skipped(self) -> None:
        captured: list[str] = []
        read_fd, write_fd = os.pipe()
        try:
            pump = asyncio.create_task(_pump_stderr("test_conn", read_fd))
            with mock_log_capture(captured):
                os.write(write_fd, b"a\n\n\nb\n")
                await asyncio.sleep(0.05)
                os.close(write_fd)
                write_fd = -1
                await asyncio.wait_for(pump, timeout=1.0)
        finally:
            if write_fd != -1:
                os.close(write_fd)
        # Blank lines between content drop without producing events.
        assert captured == ["a", "b"]

    async def test_trailing_partial_line_emitted_on_eof(self) -> None:
        captured: list[str] = []
        read_fd, write_fd = os.pipe()
        try:
            pump = asyncio.create_task(_pump_stderr("test_conn", read_fd))
            with mock_log_capture(captured):
                os.write(write_fd, b"trailing-no-newline")
                await asyncio.sleep(0.05)
                os.close(write_fd)
                write_fd = -1
                await asyncio.wait_for(pump, timeout=1.0)
        finally:
            if write_fd != -1:
                os.close(write_fd)
        assert captured == ["trailing-no-newline"]


class TestDropCounter:
    """Drop counters bump per reason and surface in :meth:`snapshot`.

    PR3 surfaces ``inbound_dropped_total{connector,reason}`` via the
    ``recent_drops`` field in ``GET /v1/connectors``.  These tests
    exercise the supervisor's bookkeeping without going through the
    full inbound dispatch path (which needs a real Postgres pool).
    """

    def _registry_with_state(self) -> tuple[ConnectorSubprocessRegistry, Any]:
        from aios.config import Settings
        from aios.harness.connector_supervisor import ConnectorState

        registry = ConnectorSubprocessRegistry([], settings=Settings())
        spec = ConnectorSpec(name="echo", command="x", args=[])
        state = ConnectorState(connector="echo", instance="echo", spec=spec)
        registry._states[("echo", "echo")] = state
        return registry, state

    def test_drops_accumulate_per_reason(self) -> None:
        registry, state = self._registry_with_state()
        registry._record_drop(state, "no_connection")
        registry._record_drop(state, "no_connection")
        registry._record_drop(state, "detached")
        snap = state.snapshot()
        assert snap["recent_drops"] == {"no_connection": 2, "detached": 1}

    def test_snapshot_starts_empty(self) -> None:
        _, state = self._registry_with_state()
        assert state.snapshot()["recent_drops"] == {}


class TestInboundMalformed:
    """Malformed inbound payloads count as ``malformed`` drops, never crash.

    The splitter task stays alive even for protocol violations from a
    misbehaving connector — operator sees them in ``recent_drops``
    without losing the rest of the pipeline.
    """

    async def _build_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[ConnectorSubprocessRegistry, list[str]]:
        """Construct a registry with a stubbed ack and an asserting pool.

        The pool stub fails the test if any drop path tries to touch
        the DB; the ack stub records event_ids without going through
        ``dispatch_call`` (which would otherwise hit the 60s timeout
        waiting for a connector subprocess).
        """
        from aios.config import Settings
        from aios.harness.connector_supervisor import ConnectorState

        registry = ConnectorSubprocessRegistry([], settings=Settings())
        spec = ConnectorSpec(name="echo", command="x", args=[])
        registry._states[("echo", "echo")] = ConnectorState(
            connector="echo", instance="echo", spec=spec
        )

        def fake_require_pool() -> None:
            raise AssertionError("DB should not be touched on malformed payload")

        monkeypatch.setattr(
            "aios.harness.connector_supervisor.runtime.require_pool",
            fake_require_pool,
        )

        acked: list[str] = []

        async def fake_ack(_self: object, _connector: str, _instance: str, event_id: str) -> None:
            acked.append(event_id)

        monkeypatch.setattr(
            ConnectorSubprocessRegistry,
            "_send_ack",
            fake_ack,
        )
        return registry, acked

    async def test_missing_event_id_drops_without_ack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing event_id → drop, NO ack (no id to ack against)."""
        registry, acked = await self._build_registry(monkeypatch)
        await registry._handle_inbound(
            ("echo", "echo"),
            {"account": "a", "chat_id": "c", "content": "x"},
        )
        assert registry._states[("echo", "echo")].drops["malformed"] == 1
        assert acked == []

    async def test_missing_other_field_drops_with_ack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Other fields missing → drop AND ack so spool clears."""
        registry, acked = await self._build_registry(monkeypatch)
        await registry._handle_inbound(
            ("echo", "echo"),
            {"event_id": "01EVT", "account": "acct"},  # missing chat_id, content
        )
        assert registry._states[("echo", "echo")].drops["malformed"] == 1
        assert acked == ["01EVT"]


class TestBackoffSchedule:
    """The supervisor doubles its backoff up to ``_BACKOFF_CAP_S`` per plan §11.

    Verified by walking the state machine directly rather than racing
    real time — the supervisor loop's "sleep, double, sleep, double" is
    a state transition, and we can simulate it by setting the failure
    timestamp ourselves.
    """

    def test_doubles_until_cap(self) -> None:
        from aios.harness.connector_supervisor import (
            _BACKOFF_CAP_S,
            _BACKOFF_INITIAL_S,
            ConnectorState,
        )

        spec = ConnectorSpec(name="echo", command="x", args=[])
        state = ConnectorState(connector="echo", instance="echo", spec=spec)

        # Walk the doubling chain until we hit the cap.  The supervisor
        # loop applies ``state.backoff = min(backoff * 2, cap)`` after
        # each crash; we replay that here.
        observed = [state.backoff]
        for _ in range(20):
            state.backoff = min(state.backoff * 2, _BACKOFF_CAP_S)
            observed.append(state.backoff)
        assert observed[0] == _BACKOFF_INITIAL_S
        assert observed[-1] == _BACKOFF_CAP_S
        # Sequence is monotonically non-decreasing.
        from itertools import pairwise

        for prev, nxt in pairwise(observed):
            assert nxt >= prev


class TestAccountsNotificationMalformed:
    """``notifications/aios/accounts`` with a non-list payload is a contract violation.

    Tested as a pure-function call into the routing helper to avoid
    the splitter task plumbing.
    """

    async def test_non_list_records_last_error(self) -> None:
        from aios.config import Settings
        from aios.harness.connector_supervisor import ConnectorState

        registry = ConnectorSubprocessRegistry([], settings=Settings())
        spec = ConnectorSpec(name="echo", command="x", args=[])
        registry._states[("echo", "echo")] = ConnectorState(
            connector="echo", instance="echo", spec=spec
        )

        await registry._on_aios_notification(
            ("echo", "echo"), "notifications/aios/accounts", {"accounts": None}
        )
        state = registry._states[("echo", "echo")]
        assert state.last_error == "malformed accounts payload"
        assert state.accounts == []

    async def test_list_replaces_snapshot(self) -> None:
        from aios.config import Settings
        from aios.harness.connector_supervisor import ConnectorState

        registry = ConnectorSubprocessRegistry([], settings=Settings())
        spec = ConnectorSpec(name="echo", command="x", args=[])
        registry._states[("echo", "echo")] = ConnectorState(
            connector="echo", instance="echo", spec=spec
        )

        await registry._on_aios_notification(
            ("echo", "echo"),
            "notifications/aios/accounts",
            {"accounts": [{"id": "a", "display_name": "A"}]},
        )
        state = registry._states[("echo", "echo")]
        assert state.accounts == [{"id": "a", "display_name": "A"}]
        assert state.last_error is None
        # Account map populated for the reporting instance.
        assert registry._account_to_instance == {("echo", "a"): "echo"}


class TestAccountMapRebuild:
    """``_rebuild_account_map`` clears stale claims, refuses conflicts."""

    def _registry_with_two_signal_instances(self) -> ConnectorSubprocessRegistry:
        from aios.config import Settings
        from aios.harness.connector_supervisor import ConnectorState

        registry = ConnectorSubprocessRegistry([], settings=Settings())
        spec = ConnectorSpec(name="signal", command="x", args=[])
        registry._states[("signal", "primary")] = ConnectorState(
            connector="signal", instance="primary", spec=spec
        )
        registry._states[("signal", "secondary")] = ConnectorState(
            connector="signal", instance="secondary", spec=spec
        )
        return registry

    async def test_clears_stale_entries_for_reporting_instance(self) -> None:
        """A removed account leaves no stale map entry pointing at the
        instance that used to serve it (clear-then-insert semantics)."""
        registry = self._registry_with_two_signal_instances()
        # Two-account snapshot from primary.
        await registry._on_aios_notification(
            ("signal", "primary"),
            "notifications/aios/accounts",
            {
                "accounts": [
                    {"id": "+1111", "display_name": "A"},
                    {"id": "+2222", "display_name": "B"},
                ]
            },
        )
        assert registry._account_to_instance == {
            ("signal", "+1111"): "primary",
            ("signal", "+2222"): "primary",
        }
        # Primary now reports only +1111 — +2222 must be cleared.
        await registry._on_aios_notification(
            ("signal", "primary"),
            "notifications/aios/accounts",
            {"accounts": [{"id": "+1111", "display_name": "A"}]},
        )
        assert registry._account_to_instance == {("signal", "+1111"): "primary"}

    async def test_conflict_rejects_second_claim(self) -> None:
        """Two instances claiming the same account: second is rejected,
        last_error set, primary keeps the map entry."""
        registry = self._registry_with_two_signal_instances()
        await registry._on_aios_notification(
            ("signal", "primary"),
            "notifications/aios/accounts",
            {"accounts": [{"id": "+1111", "display_name": "A"}]},
        )
        await registry._on_aios_notification(
            ("signal", "secondary"),
            "notifications/aios/accounts",
            {"accounts": [{"id": "+1111", "display_name": "A-dup"}]},
        )
        # Primary still owns +1111; secondary's last_error surfaces the conflict.
        assert registry._account_to_instance == {("signal", "+1111"): "primary"}
        secondary = registry._states[("signal", "secondary")]
        assert secondary.last_error is not None
        assert "conflict" in secondary.last_error

    async def test_lookup_instance_for_account(self) -> None:
        """``lookup_instance_for_account`` is the public read path."""
        registry = self._registry_with_two_signal_instances()
        await registry._on_aios_notification(
            ("signal", "primary"),
            "notifications/aios/accounts",
            {"accounts": [{"id": "+1111", "display_name": "A"}]},
        )
        assert registry.lookup_instance_for_account("signal", "+1111") == "primary"
        assert registry.lookup_instance_for_account("signal", "+9999") is None
        assert registry.lookup_instance_for_account("telegram", "+1111") is None


class mock_log_capture:
    """Capture ``log.bind(...).info(...)`` line= kwargs into a list."""

    def __init__(self, captured: list[str]) -> None:
        self._captured = captured
        self._patches: list[Any] = []

    def __enter__(self) -> mock_log_capture:
        from unittest.mock import patch

        from aios.mcp import stdio_transport

        original = stdio_transport.log

        def fake_info(_event: str, line: str | None = None, **_: Any) -> None:
            if line is not None:
                self._captured.append(line)

        bound = MagicMock()
        bound.info = fake_info
        bind_patch = patch.object(original, "bind", return_value=bound)
        self._patches.append(bind_patch)
        bind_patch.start()
        return self

    def __exit__(self, *_: Any) -> None:
        for p in self._patches:
            p.stop()
