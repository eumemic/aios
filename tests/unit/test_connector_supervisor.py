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

    def test_passes_when_aios_connector_key_present(self) -> None:
        registry = self._registry()
        init = _fake_init_result({_AIOS_EXPERIMENTAL_KEY: {}})
        # Should not raise.
        registry._validate_capability("echo", init)

    def test_raises_on_missing_experimental_dict(self) -> None:
        registry = self._registry()
        init = _fake_init_result(None)
        with pytest.raises(RuntimeError, match=r"experimental\..*aios/connector"):
            registry._validate_capability("echo", init)

    def test_raises_on_empty_experimental_dict(self) -> None:
        registry = self._registry()
        init = _fake_init_result({})
        with pytest.raises(RuntimeError, match=r"experimental\..*aios/connector"):
            registry._validate_capability("echo", init)

    def test_raises_when_other_experimental_keys_present(self) -> None:
        registry = self._registry()
        init = _fake_init_result({"some/other/cap": {"option": True}})
        with pytest.raises(RuntimeError, match=r"experimental\..*aios/connector"):
            registry._validate_capability("echo", init)


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

    def test_returns_spec_with_default_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        assert specs[0].name == "echo"
        # Plan #11: cwd defaults to ``connectors_dir / name`` when the
        # factory leaves it None.
        assert specs[0].cwd == connectors_dir / "echo"

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
        assert specs[0].cwd == explicit


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
