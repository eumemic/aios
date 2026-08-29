"""``browser-cli`` — the stdlib-only client the worker execs inside the browser
container as ``browser-cli invoke '<request JSON>'``.

Contract (``browser_protocol``): stdout is exactly one JSON response document,
and the process exits 0 IFF it produced one — INCLUDING an ``ok: false``
document the daemon chose to return. A nonzero exit or unparseable stdout is a
transport/daemon failure the worker surfaces as "browser unavailable".

So this client forwards the request bytes VERBATIM (a malformed request is the
daemon's to reject with ``invalid_request`` at exit 0, not ours), and reserves
nonzero exits for genuine transport faults: cannot connect (7), read failed
(8), unparseable reply (9). It must never exit 137 — that is the exec wrapper's
SIGKILL, read upstream as a timeout — so every wait is bounded below that.
"""

from __future__ import annotations

import contextlib
import json
import socket
import sys
import time

from aios_browser_driver.sockpath import socket_path

# The warm path is a liveness probe only: an invoke can arrive the instant
# ``docker run`` returns, before the daemon has bound the socket. Retry the
# connect with backoff; the daemon binds before launching the browser, so the
# window is short.
_CONNECT_BACKOFF_START_S = 0.025
_CONNECT_BACKOFF_MAX_S = 0.25
# The daemon self-reports (e.g. ``action_timeout``) at ``timeout_ms``; give it
# that plus transit before we give up, staying under the exec wrapper's
# ``timeout_ms + 5s`` SIGKILL.
_RESPONSE_GRACE_S = 3.0
# Fallback only, for a payload missing/garbling ``timeout_ms``; the worker
# always sends one. Mirrors ``BrowserRequest.timeout_ms``'s default — not
# imported, to keep this exec-hot-path client pydantic-free.
_DEFAULT_TIMEOUT_MS = 30_000
_READ_CHUNK = 65536

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONNECT = 7
EXIT_READ = 8
EXIT_PARSE = 9


def _deadline_from(payload: str) -> float:
    try:
        timeout_ms = int(json.loads(payload)["timeout_ms"])
    except (ValueError, TypeError, KeyError):
        timeout_ms = _DEFAULT_TIMEOUT_MS
    return time.monotonic() + timeout_ms / 1000.0 + _RESPONSE_GRACE_S


def _connect(deadline: float) -> socket.socket | None:
    path = socket_path()
    backoff = _CONNECT_BACKOFF_START_S
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            time.sleep(backoff)
            backoff = min(backoff * 2, _CONNECT_BACKOFF_MAX_S)
        except OSError:
            sock.close()
            return None
    return None


def _read_line(sock: socket.socket, deadline: float) -> bytes | None:
    chunks: list[bytes] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        sock.settimeout(remaining)
        try:
            chunk = sock.recv(_READ_CHUNK)
        except OSError:
            return None
        if not chunk:
            break  # EOF
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    data = b"".join(chunks)
    return data or None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 3 or argv[1] != "invoke":
        sys.stderr.write("usage: browser-cli invoke '<request JSON>'\n")
        return EXIT_USAGE
    payload = argv[2]
    deadline = _deadline_from(payload)

    sock = _connect(deadline)
    if sock is None:
        sys.stderr.write("browser-cli: driver socket unavailable\n")
        return EXIT_CONNECT
    try:
        sock.sendall(payload.encode("utf-8") + b"\n")
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_WR)
        line = _read_line(sock, deadline)
    finally:
        sock.close()

    if line is None:
        sys.stderr.write("browser-cli: no reply from driver\n")
        return EXIT_READ
    text = line.decode("utf-8", "replace").strip()
    try:
        json.loads(text)
    except ValueError:
        sys.stderr.write("browser-cli: unparseable driver reply\n")
        return EXIT_PARSE
    sys.stdout.write(text + "\n")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
