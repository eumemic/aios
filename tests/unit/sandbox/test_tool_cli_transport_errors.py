"""``bin/tool`` writes a parseable JSON error envelope on transport failure.

Issue #2227: when the CLI's own request to the broker timed out, it
emitted a Python traceback (``TimeoutError: timed out``) with nothing on
stdout. Every caller ``json.loads``-es stdout, so a timeout surfaced as
``Expecting value: line 1 column 1`` — indistinguishable from a
malformed response, and one ``except Exception: rows = []`` away from
being read as "zero results".

Contract pinned here: on timeout or any transport-level failure, stdout
carries ``{"error": {"kind": ..., "timeout_s": ..., "path": ...}}`` and
the process exits non-zero; diagnostics go to stderr. A successful call
(including a genuinely empty result set) is unchanged.

The tests drive real sockets (a deliberately slow local HTTP server and
a slow UDS server) rather than mocking ``urlopen`` so the exception
types are the ones the stdlib actually raises.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

_SECRET = "s3cret-token"


def _make_handler(
    delay_s: float, body: bytes, post_delay_s: float | None, post_body: bytes | None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, delay_s: float = delay_s, body: bytes = body) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            if delay_s:
                time.sleep(delay_s)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

        def do_GET(self) -> None:
            self._respond()

        def do_POST(self) -> None:
            self._respond(
                delay_s if post_delay_s is None else post_delay_s,
                body if post_body is None else post_body,
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


@contextmanager
def _http_server(
    delay_s: float,
    body: bytes,
    *,
    post_delay_s: float | None = None,
    post_body: bytes | None = None,
) -> Iterator[str]:
    """Local broker stand-in; POST may differ from GET in delay/body."""
    handler = _make_handler(delay_s, body, post_delay_s, post_body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host!s}:{port}"
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def _slow_uds_server(delay_s: float) -> Iterator[str]:
    """A UDS listener that accepts and reads but never answers in time."""
    tmpdir = tempfile.mkdtemp(prefix="toolcli")
    path = os.path.join(tmpdir, "b.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(4)
    stop = threading.Event()

    def serve() -> None:
        srv.settimeout(0.1)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            try:
                conn.recv(65536)
                stop.wait(delay_s)
            finally:
                conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"unix://{path}"
    finally:
        stop.set()
        thread.join(timeout=2)
        srv.close()
        Path(path).unlink(missing_ok=True)
        os.rmdir(tmpdir)


def _run(
    tool_module: ModuleType,
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    code = 0
    try:
        tool_module.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tool_module: ModuleType) -> pytest.MonkeyPatch:
    monkeypatch.setenv("TOOL_BROKER_SECRET", _SECRET)
    monkeypatch.delenv("AIOS_TRIGGER_OBSERVATION", raising=False)
    monkeypatch.setattr(tool_module, "_REQUEST_TIMEOUT_S", 0.3)
    return monkeypatch


class TestTimeoutEnvelope:
    def test_http_timeout_writes_json_envelope_and_exits_nonzero(
        self,
        tool_module: ModuleType,
        env: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with _http_server(delay_s=1.0, body=b"{}") as url:
            env.setenv("TOOL_BROKER_URL", url)
            code, out, err = _run(tool_module, ["tool"], capsys)

        assert code != 0
        envelope = json.loads(out)
        assert envelope == {"error": {"kind": "timeout", "timeout_s": 0.3, "path": "/tools"}}
        # Diagnostics on stderr, never the secret on stdout.
        assert "timed out" in err
        assert _SECRET not in out

    def test_uds_timeout_writes_json_envelope_and_exits_nonzero(
        self,
        tool_module: ModuleType,
        env: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with _slow_uds_server(delay_s=1.0) as url:
            env.setenv("TOOL_BROKER_URL", url)
            code, out, _err = _run(tool_module, ["tool"], capsys)

        assert code != 0
        envelope = json.loads(out)
        assert envelope["error"]["kind"] == "timeout"
        assert envelope["error"]["timeout_s"] == 0.3
        assert envelope["error"]["path"] == "/tools"

    def test_invocation_timeout_names_invocation_path(
        self,
        tool_module: ModuleType,
        env: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Discovery succeeds fast; the invocation itself is the slow path.
        listing = json.dumps({"builtins": [{"name": "http_request"}], "servers": []})
        with _http_server(0, listing.encode(), post_delay_s=1.0, post_body=b"{}") as url:
            env.setenv("TOOL_BROKER_URL", url)
            code, out, _err = _run(tool_module, ["tool", "http_request", "{}"], capsys)

        assert code != 0
        envelope = json.loads(out)
        assert envelope["error"]["kind"] == "timeout"
        assert envelope["error"]["path"] == "/builtins/http_request"


class TestTransportFailureEnvelope:
    def test_unreachable_broker_writes_json_envelope(
        self,
        tool_module: ModuleType,
        env: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Bind then close to get a port with nothing listening.
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        env.setenv("TOOL_BROKER_URL", f"http://127.0.0.1:{port}")
        code, out, err = _run(tool_module, ["tool"], capsys)

        assert code == 2
        envelope = json.loads(out)
        assert envelope["error"]["kind"] == "unreachable"
        assert envelope["error"]["path"] == "/tools"
        assert "broker unreachable" in err

    def test_missing_uds_socket_writes_json_envelope(
        self,
        tool_module: ModuleType,
        env: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        env.setenv("TOOL_BROKER_URL", f"unix://{tmp_path / 'nope.sock'}")
        code, out, _err = _run(tool_module, ["tool"], capsys)

        assert code == 2
        envelope = json.loads(out)
        assert envelope["error"]["kind"] == "unreachable"

    def test_malformed_success_body_writes_json_envelope(
        self,
        tool_module: ModuleType,
        env: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with _http_server(delay_s=0, body=b"<html>not json</html>") as url:
            env.setenv("TOOL_BROKER_URL", url)
            code, out, _err = _run(tool_module, ["tool"], capsys)

        assert code == 2
        envelope = json.loads(out)
        assert envelope["error"]["kind"] == "malformed_response"


class TestSuccessUnchanged:
    def test_empty_result_set_is_distinguishable_from_timeout(
        self,
        tool_module: ModuleType,
        env: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        listing = json.dumps({"builtins": [{"name": "http_request"}], "servers": []})
        empty = json.dumps({"content": json.dumps({"data": []})})
        with _http_server(0, listing.encode(), post_body=empty.encode()) as url:
            env.setenv("TOOL_BROKER_URL", url)
            code, out, err = _run(tool_module, ["tool", "http_request", "{}"], capsys)

        assert code == 0
        assert json.loads(out) == {"data": []}
        assert err == ""

    def test_successful_listing_unchanged(
        self,
        tool_module: ModuleType,
        env: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        listing = json.dumps(
            {"builtins": [{"name": "web_fetch", "description": "Fetch."}], "servers": []}
        )
        with _http_server(delay_s=0, body=listing.encode()) as url:
            env.setenv("TOOL_BROKER_URL", url)
            code, out, _err = _run(tool_module, ["tool"], capsys)

        assert code == 0
        assert "web_fetch" in out
        assert '"error"' not in out
