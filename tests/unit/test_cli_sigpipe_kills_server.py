"""End-to-end pinning of the client-vs-server SIGPIPE contract.

Server processes (``aios api`` / ``aios worker``) must keep Python's
default ``SIG_IGN``.  Client subcommands install ``SIG_DFL`` inside
``run_or_die`` to preserve ``yes | head`` exit-141 semantics.  The
boot-import side and AST guard live in ``test_no_silent_kill_handlers.py``;
this file's job is the runtime behavior at the broken-socket boundary.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap


def _run_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_broken_pipe_after_server_import_raises_not_kills() -> None:
    """A subprocess on the server import path writes to a broken socket
    and surfaces ``BrokenPipeError`` instead of being SIGPIPE-killed.
    This is the inverted shape of the original silent-worker-exit
    repro."""
    result = _run_subprocess("""
        import socket
        import sys
        import aios.cli.app  # noqa: F401

        a, b = socket.socketpair()
        b.close()
        try:
            for _ in range(10000):
                a.send(b"x" * 4096)
        except BrokenPipeError:
            print("BROKEN_PIPE_RAISED")
            sys.exit(0)
        sys.exit(0)
    """)
    assert result.returncode == 0, (
        f"server import path armed SIGPIPE-kill. "
        f"rc={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
    assert result.stdout.strip() == "BROKEN_PIPE_RAISED"


def test_broken_pipe_inside_run_or_die_kills_with_sigpipe() -> None:
    """Inside ``run_or_die``'s scope, a broken-socket write is
    SIGPIPE-killed (rc == -SIGPIPE).  Preserves the standard UNIX
    broken-pipe contract for client commands."""
    result = _run_subprocess("""
        import socket
        from aios.cli.runtime import run_or_die

        def fn() -> None:
            a, b = socket.socketpair()
            b.close()
            for _ in range(10000):
                a.send(b"x" * 4096)

        run_or_die(fn)
    """)
    assert result.returncode == -signal.SIGPIPE, (
        f"expected rc=-{int(signal.SIGPIPE)}; "
        f"got rc={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
    assert "Traceback" not in result.stderr
