"""Exception hierarchy for aios-signal."""

from __future__ import annotations

from typing import Any


class SignalConnectorError(Exception):
    """Base class for all aios-signal errors."""


class RpcError(SignalConnectorError):
    """JSON-RPC error from signal-cli.

    ``code`` / ``data`` carry the structured fields when present —
    captcha-required puts the challenge in ``data`` and would be lost
    on a string flatten.
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
