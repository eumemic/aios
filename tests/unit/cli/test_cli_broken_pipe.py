"""Regression coverage for issue #116.

When the stdout pipe breaks while the CLI is writing a response (e.g. a
downstream pager or scripted consumer exits early), ``run_or_die`` must
translate the resulting ``BrokenPipeError`` into a clean exit instead of
a traceback.  This keeps mutating verbs from looking like they failed
when in fact the server-side write already landed.
"""

from __future__ import annotations

import io
import signal
import sys
from collections.abc import Iterator

import pytest
import typer

from aios.cli.runtime import run_or_die


@pytest.fixture(autouse=True)
def _isolate_sigpipe_handler() -> Iterator[None]:
    """Force ``SIG_IGN`` before each test and restore after — earlier
    tests (in this file or elsewhere) can leak ``SIG_DFL`` from
    ``run_or_die`` into our pre-state."""
    if not hasattr(signal, "SIGPIPE"):
        yield
        return
    saved = signal.getsignal(signal.SIGPIPE)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGPIPE, saved)


def test_run_or_die_swallows_brokenpipe_and_exits_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``fn`` raises ``BrokenPipeError`` directly, we exit 0 quietly.

    Swaps ``sys.stdout`` to an in-memory stream first so the silence-
    stdout path has a non-__stdout__ object to replace, mirroring the
    real-runtime shape where the broken stdout is a pipe-backed dup of
    fd 1 rather than the interpreter's own stdout.
    """
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    def fn() -> int | None:
        raise BrokenPipeError(32, "Broken pipe")

    with pytest.raises(typer.Exit) as excinfo:
        run_or_die(fn)

    assert excinfo.value.exit_code == 0


def test_run_or_die_swallows_brokenpipe_raised_from_stdout_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broken pipe raised from a buffered ``stdout.write`` inside the command
    still resolves to a clean exit — the typical real-world shape where
    ``json`` rendering fails mid-flush after the HTTP call succeeded."""

    class BrokenStream(io.StringIO):
        def write(self, _s: str) -> int:  # type: ignore[override]
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(sys, "stdout", BrokenStream())

    def fn() -> int | None:
        # Simulates the post-HTTP rendering step that fails when the
        # consumer closed its end of the pipe.
        sys.stdout.write('{"ok": true}\n')
        return None

    with pytest.raises(typer.Exit) as excinfo:
        run_or_die(fn)

    assert excinfo.value.exit_code == 0


def test_run_or_die_installs_sig_dfl_for_client_commands() -> None:
    """``run_or_die`` is the universal wrapper for client subcommands; it
    installs ``SIG_DFL`` for SIGPIPE so ``aios sessions stream | head``
    terminates with the standard UNIX exit-141 contract instead of a
    Python ``BrokenPipeError`` traceback.  Server commands (``aios api``,
    ``aios worker``) do NOT route through ``run_or_die`` and therefore
    keep Python's default ``SIG_IGN`` — broken upstream sockets surface
    as ``BrokenPipeError`` rather than killing the long-running process.
    """
    if not hasattr(signal, "SIGPIPE"):
        pytest.skip("SIGPIPE not available on this platform")

    # Sanity check: the import of aios.cli.app no longer mutates SIGPIPE.
    # The autouse fixture restored the handler before this test ran.
    assert signal.getsignal(signal.SIGPIPE) == signal.SIG_IGN

    run_or_die(lambda: None)

    assert signal.getsignal(signal.SIGPIPE) == signal.SIG_DFL
