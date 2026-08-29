"""``browser-cli`` forwards the request verbatim and maps outcomes to the
exit-code contract: 0 with the reply on stdout when the daemon answered
(including ``ok:false``), and a nonzero transport code (7 connect / 8 read /
9 parse) otherwise — never 137."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import tempfile
import threading
from collections.abc import Callable, Iterator

import pytest
from aios_browser_driver import cli


def _read_line(conn: socket.socket) -> bytes:
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf


@contextlib.contextmanager
def fake_server(behavior: Callable[[socket.socket], None]) -> Iterator[str]:
    tmpdir = tempfile.mkdtemp()
    sock_path = os.path.join(tmpdir, "driver.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    def _serve() -> None:
        with contextlib.suppress(OSError):
            conn, _ = srv.accept()
            with conn:
                behavior(conn)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        yield sock_path
    finally:
        srv.close()
        thread.join(timeout=2)
        with contextlib.suppress(OSError):
            os.remove(sock_path)
        os.rmdir(tmpdir)


@pytest.fixture(autouse=True)
def _fast_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the connect/read deadline short so the failure tests are fast.
    monkeypatch.setattr(cli, "_RESPONSE_GRACE_S", 0.2)


_REQUEST = json.dumps({"op": "status", "timeout_ms": 500})


def test_reply_is_echoed_to_stdout_at_exit_0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    canned = b'{"ok": true, "boot": "01BOOT", "epoch": 0}\n'

    def behavior(conn: socket.socket) -> None:
        _read_line(conn)
        conn.sendall(canned)

    with fake_server(behavior) as sock_path:
        monkeypatch.setenv("AIOS_BROWSER_DRIVER_SOCK", sock_path)
        rc = cli.main(["browser-cli", "invoke", _REQUEST])
    assert rc == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"ok": True, "boot": "01BOOT", "epoch": 0}


def test_request_is_forwarded_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bytes] = []

    def behavior(conn: socket.socket) -> None:
        seen.append(_read_line(conn))
        conn.sendall(b'{"ok": false, "boot": "b", "epoch": 0}\n')

    malformed = "{ this is the daemon's to reject "
    with fake_server(behavior) as sock_path:
        monkeypatch.setenv("AIOS_BROWSER_DRIVER_SOCK", sock_path)
        rc = cli.main(["browser-cli", "invoke", malformed])
    assert rc == cli.EXIT_OK  # ok:false still exits 0
    assert seen[0].rstrip(b"\n") == malformed.encode()


def test_missing_socket_is_exit_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_BROWSER_DRIVER_SOCK", "/nonexistent/aios-driver.sock")
    assert cli.main(["browser-cli", "invoke", _REQUEST]) == cli.EXIT_CONNECT


def test_eof_without_reply_is_exit_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def behavior(conn: socket.socket) -> None:
        _read_line(conn)  # read then close without answering

    with fake_server(behavior) as sock_path:
        monkeypatch.setenv("AIOS_BROWSER_DRIVER_SOCK", sock_path)
        assert cli.main(["browser-cli", "invoke", _REQUEST]) == cli.EXIT_READ


def test_unparseable_reply_is_exit_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    def behavior(conn: socket.socket) -> None:
        _read_line(conn)
        conn.sendall(b"not json at all\n")

    with fake_server(behavior) as sock_path:
        monkeypatch.setenv("AIOS_BROWSER_DRIVER_SOCK", sock_path)
        assert cli.main(["browser-cli", "invoke", _REQUEST]) == cli.EXIT_PARSE


def test_bad_argv_is_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli.main(["browser-cli"]) != cli.EXIT_OK
    assert cli.main(["browser-cli", "frobnicate", "{}"]) != cli.EXIT_OK
