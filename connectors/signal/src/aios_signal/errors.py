"""Exception hierarchy for aios-signal."""

from __future__ import annotations

from typing import Any


class SignalConnectorError(Exception):
    """Base class for all aios-signal errors."""


class RpcError(SignalConnectorError):
    """JSON-RPC error from signal-cli.

    Carries the structured ``code`` and ``data`` fields when present —
    captcha-required errors put the challenge URL inside ``data`` and
    would be lost if we flattened to a string.  Either field can be
    ``None`` when the daemon returned a non-structured failure (e.g.
    connection drop before a response).
    """

    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class RpcTimeoutError(RpcError):
    pass


class ListenerClosedError(RpcError):
    pass


class DaemonCrashError(SignalConnectorError):
    pass


class BotAccountNotFoundError(SignalConnectorError):
    pass
