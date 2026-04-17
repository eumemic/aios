"""Exception hierarchy for aios-signal."""

from __future__ import annotations


class SignalConnectorError(Exception):
    """Base class for all aios-signal errors."""


class RpcError(SignalConnectorError):
    pass


class RpcTimeoutError(RpcError):
    pass


class ListenerClosedError(RpcError):
    pass


class DaemonCrashError(SignalConnectorError):
    pass


class BotAccountNotFoundError(SignalConnectorError):
    pass
