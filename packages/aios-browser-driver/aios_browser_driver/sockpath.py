"""The AF_UNIX socket both halves use: the daemon binds it, ``browser-cli``
connects to it.

Deliberately NOT under ``/workspace`` (the account plane) — the socket is
ephemeral runtime state and must never be reachable through any plane-tailing
API path. The env override exists only as a test seam; production never sets it
(the browser spec pins ``environment={}``), so the default path is authoritative
in every real deployment.
"""

from __future__ import annotations

import os

_DEFAULT_SOCKET_PATH = "/run/aios/driver.sock"
_SOCKET_PATH_ENV = "AIOS_BROWSER_DRIVER_SOCK"


def socket_path() -> str:
    return os.environ.get(_SOCKET_PATH_ENV) or _DEFAULT_SOCKET_PATH
