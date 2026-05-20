"""Unit tests for the Unix-domain-socket transport on :class:`McpBroker`.

The TCP path is covered by ``test_mcp_proxy.py`` and the request envelope
contract by ``test_broker_http_contract.py``. These tests pin the
lifecycle of the socket file itself — creation, permissions, cleanup,
stale-overwrite — plus the dual-listen behavior when ``socket_path`` is
set.
"""

from __future__ import annotations

import socket
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from aios.sandbox.mcp_proxy import McpBroker


@pytest.fixture
def short_tmp_path() -> Iterator[Path]:
    """A short-path tmp dir for UDS bindings.

    macOS limits ``AF_UNIX`` paths to ~104 chars; pytest's ``tmp_path``
    routes through ``/private/var/folders/...`` and overshoots that.
    ``tempfile.mkdtemp()`` under ``/tmp`` keeps us comfortably under
    the limit on every platform we care about.
    """
    import shutil

    d = Path(tempfile.mkdtemp(prefix="aios-uds-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class TestUdsLifecycle:
    async def test_broker_creates_socket_file_on_start(self, short_tmp_path: Path) -> None:
        sock_path = short_tmp_path / "b.sock"
        broker = McpBroker(socket_path=sock_path)
        await broker.start()
        try:
            assert sock_path.exists()
            assert stat.S_ISSOCK(sock_path.stat().st_mode)
        finally:
            await broker.stop()

    async def test_broker_removes_socket_file_on_stop(self, short_tmp_path: Path) -> None:
        sock_path = short_tmp_path / "b.sock"
        broker = McpBroker(socket_path=sock_path)
        await broker.start()
        assert sock_path.exists()
        await broker.stop()
        assert not sock_path.exists()

    async def test_broker_socket_has_world_writable_permissions(self, short_tmp_path: Path) -> None:
        """Sandbox container's non-root user must be able to write to the
        socket. ``chmod 0o666`` is the simplest portable answer."""
        sock_path = short_tmp_path / "b.sock"
        broker = McpBroker(socket_path=sock_path)
        await broker.start()
        try:
            mode = sock_path.stat().st_mode & 0o777
            assert mode == 0o666
        finally:
            await broker.stop()

    async def test_broker_overwrites_stale_socket_file(self, short_tmp_path: Path) -> None:
        """If a previous worker crashed without cleaning up, the new
        broker must clear the stale file rather than fail to bind."""
        sock_path = short_tmp_path / "b.sock"
        sock_path.write_bytes(b"stale")
        assert sock_path.exists()

        broker = McpBroker(socket_path=sock_path)
        await broker.start()
        try:
            assert sock_path.exists()
            assert stat.S_ISSOCK(sock_path.stat().st_mode)
        finally:
            await broker.stop()

    async def test_broker_socket_path_in_parent_that_does_not_exist(
        self, short_tmp_path: Path
    ) -> None:
        """The host parent dir (e.g. ``/var/run/aios``) may not exist at
        worker start — broker creates it."""
        sock_path = short_tmp_path / "d" / "n" / "b.sock"
        assert not sock_path.parent.exists()
        broker = McpBroker(socket_path=sock_path)
        await broker.start()
        try:
            assert sock_path.exists()
        finally:
            await broker.stop()


class TestDualListen:
    async def test_dual_listens_tcp_and_uds_when_socket_path_set(
        self, short_tmp_path: Path
    ) -> None:
        """When socket_path is set, both transports must accept connections."""
        sock_path = short_tmp_path / "b.sock"
        broker = McpBroker(socket_path=sock_path)
        await broker.start()
        try:
            # TCP still bound on an ephemeral port.
            tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp.settimeout(2.0)
            tcp.connect(("127.0.0.1", broker.port))
            tcp.close()

            # UDS also accepting connections.
            uds = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            uds.settimeout(2.0)
            uds.connect(str(sock_path))
            uds.close()
        finally:
            await broker.stop()

    async def test_tcp_only_when_socket_path_none(self) -> None:
        """No socket_path → broker.socket_path is None and only TCP is up."""
        broker = McpBroker()
        await broker.start()
        try:
            assert broker.socket_path is None
            tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp.settimeout(2.0)
            tcp.connect(("127.0.0.1", broker.port))
            tcp.close()
        finally:
            await broker.stop()


class TestSocketPathProperty:
    async def test_socket_path_property_mirrors_constructor(self, short_tmp_path: Path) -> None:
        sock_path = short_tmp_path / "b.sock"
        broker = McpBroker(socket_path=sock_path)
        await broker.start()
        try:
            assert broker.socket_path == sock_path
        finally:
            await broker.stop()
