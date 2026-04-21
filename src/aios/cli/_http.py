"""Shared HTTP + env plumbing for ``aios <verb-group>`` CLI modules.

The sibling modules — ``aios.cli.connections``, ``aios.cli.bindings``,
``aios.cli.rules`` — all do the same four things around an httpx call:

1. Read ``AIOS_API_KEY`` + ``AIOS_API_URL`` (or host+port) from env.
2. Open a timed-out ``httpx.AsyncClient``.
3. Print a ``"{prog}: HTTP {code}: {text}"`` line to stderr on non-2xx.
4. Print the JSON body to stdout on success.

This module hosts the first three — the fourth is one line and stays
in the verb handlers so the success shape (array vs. single object)
stays explicit per call site.
"""

from __future__ import annotations

import os
import sys

import httpx


class CliError(Exception):
    """Raised for user-visible config or argument errors (bad env, bad JSON, etc.).

    Handler wraps this at the dispatch layer: prints the message to
    stderr and returns exit code 2 without issuing any HTTP call.
    """


def require_env(prog: str) -> tuple[str, str]:
    """Return ``(api_url, api_key)`` from env or raise :class:`CliError`.

    ``prog`` is the CLI program name (e.g. ``"aios connections"``) —
    used in the error message so operators know which verb-group
    complained.
    """
    api_key = os.environ.get("AIOS_API_KEY")
    if not api_key:
        raise CliError(f"{prog}: AIOS_API_KEY is required")
    api_url = os.environ.get(
        "AIOS_API_URL",
        f"http://{os.environ.get('AIOS_API_HOST', '127.0.0.1')}"
        f":{os.environ.get('AIOS_API_PORT', '8080')}",
    )
    return api_url, api_key


def print_http_error(prog: str, response: httpx.Response) -> None:
    """Print the standard ``"{prog}: HTTP {code}: {text}"`` line to stderr."""
    print(f"{prog}: HTTP {response.status_code}: {response.text}", file=sys.stderr)


def async_client() -> httpx.AsyncClient:
    """Return an httpx.AsyncClient with the standard CLI timeout."""
    return httpx.AsyncClient(timeout=30.0)
