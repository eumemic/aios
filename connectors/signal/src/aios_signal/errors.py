"""Exception hierarchy for aios-signal.

One shared base so operators can catch connector-specific failures without
grabbing unrelated stdlib exceptions.
"""

from __future__ import annotations


class SignalConnectorError(Exception):
    """Base class for all aios-signal errors."""


class RpcError(SignalConnectorError):
    """signal-cli JSON-RPC returned an error or the transport failed."""


class RpcTimeoutError(RpcError):
    """An RPC call exceeded its timeout."""


class ListenerClosedError(RpcError):
    """The persistent listener connection to signal-cli closed unexpectedly."""


class DaemonCrashError(SignalConnectorError):
    """The signal-cli subprocess exited unexpectedly."""


class BotAccountNotFoundError(SignalConnectorError):
    """signal-cli has no registered account matching the configured phone."""
