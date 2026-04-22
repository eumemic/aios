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

import pytest
import typer

from aios.cli.runtime import run_or_die


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


def test_sigpipe_handler_installed_at_cli_import() -> None:
    """Importing the CLI app should restore the default SIGPIPE disposition.

    Python by default turns SIGPIPE into a BrokenPipeError; that's
    helpful for handling it in Python code, but it means ``aios | head``
    leaves a BrokenPipeError traceback on stderr once the consumer
    exits.  Restoring ``SIG_DFL`` lets the kernel kill us silently on
    SIGPIPE, matching the behavior of standard UNIX tools.

    Note: signal handlers are process-wide; this test is a sanity check
    on the initial import's effect and does not guarantee isolation
    from other tests that touch SIGPIPE.  No other code in the project
    sets a SIGPIPE handler, so ordering flakes would indicate someone
    added one.
    """
    if not hasattr(signal, "SIGPIPE"):
        pytest.skip("SIGPIPE not available on this platform")

    # Import for side-effect — this should install the handler.
    import aios.cli.app  # noqa: F401

    handler = signal.getsignal(signal.SIGPIPE)
    assert handler == signal.SIG_DFL
