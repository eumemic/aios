"""Per-invocation CLI state + the error-handling shim used by every command.

Split out of :mod:`aios.cli.app` so command modules can import it without
introducing a circular dependency (the root app imports every command
module at its tail to register sub-apps).
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import typer

from aios.cli.client import AiosApiError, AiosClient
from aios.cli.files import PayloadError
from aios.cli.output import OutputFormat, print_error

if TYPE_CHECKING:
    from aios_sdk import Client


@dataclass(slots=True)
class CliState:
    """Per-invocation CLI state set from the root callback."""

    base_url: str
    api_key: str | None
    output_format: OutputFormat
    verbose: bool

    def client(self) -> AiosClient:
        return AiosClient(base_url=self.base_url, api_key=self.api_key)

    def sdk_client(self) -> Client:
        """Build a typed SDK ``Client`` from the resolved base_url + api_key.

        Used by the porcelain modules that have migrated off the
        hand-written ``AiosClient`` (``status``, ``tail``, the streaming
        subcommands of ``sessions``). The hand-written client stays for
        the remaining commands (CRUD modules, ``chat``, sessions
        ``events``) until they migrate too.

        Unlike :func:`aios_sdk.client_from_env`, this accepts a missing
        ``api_key`` and constructs a Client with an empty Bearer token —
        the ``aios status`` command needs to probe an unauthenticated
        ``/health`` and report whether ``AIOS_API_KEY`` was set, so a
        raise-on-missing surface here would be the wrong default.
        """
        from aios_sdk import Client

        return Client(base_url=self.base_url, token=self.api_key or "")


def get_state(ctx: typer.Context) -> CliState:
    """Return the :class:`CliState` attached to the typer context."""
    state = ctx.obj
    assert isinstance(state, CliState)
    return state


def run_or_die(fn: Callable[[], int | None]) -> None:
    """Execute ``fn`` and translate ``AiosApiError`` into a clean CLI exit.

    Keeps tracebacks hidden and prints the server's error envelope.
    ``BrokenPipeError`` during output rendering — raised when a downstream
    consumer (pipe, pager, script) closed its end while we were writing —
    resolves to a silent ``exit 0``; by the time we're rendering, any
    server-side mutation has already landed, and we don't want pipe
    failures to mask that (issue #116).  Non-API exceptions bubble.
    """
    try:
        rc = fn()
    except AiosApiError as exc:
        print_error(f"{exc.error_type}: {exc.message}")
        if exc.detail:
            print_error(f"detail: {exc.detail}")
        exit_code = 2 if exc.status_code == 401 else 1
        raise typer.Exit(exit_code) from exc
    except PayloadError as exc:
        print_error(str(exc))
        raise typer.Exit(64) from exc
    except BrokenPipeError:
        _silence_stdout()
        raise typer.Exit(0) from None
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        raise typer.Exit(130) from None
    if rc:
        raise typer.Exit(rc)


def _silence_stdout() -> None:
    """Swap ``sys.stdout`` for ``/dev/null`` so later writes (notably the
    interpreter-shutdown flush) don't re-raise ``BrokenPipeError`` on the
    closed pipe.

    Python 3.13 routes ``__del__`` exceptions through
    ``sys.unraisablehook`` — it no longer silently drops them as earlier
    versions did — so we also explicitly close the original stream when
    it's the real process stdout, swallowing the ``BrokenPipeError`` the
    flush inside ``close()`` will raise.  Pytest capture streams and
    test monkeypatches are left untouched so teardown can restore them.
    """
    # Kept open for the remainder of the process; closing the devnull
    # stream would make subsequent writes (or the shutdown flush) raise
    # again.
    devnull = open(os.devnull, "w")  # noqa: SIM115
    old = sys.stdout
    sys.stdout = devnull
    if sys.__stdout__ is not None and old is sys.__stdout__:
        with contextlib.suppress(Exception):
            old.close()
