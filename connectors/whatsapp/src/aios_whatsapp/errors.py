"""Exception hierarchy for aios-whatsapp."""

from __future__ import annotations

from typing import Any


class WhatsappConnectorError(Exception):
    """Base class for all aios-whatsapp errors."""


class RpcError(WhatsappConnectorError):
    """JSON-RPC error from the WhatsApp daemon.

    ``code`` / ``data`` preserve the JSON-RPC structured-error fields
    so callers can discriminate on shape rather than string-flattening.
    """

    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class RpcTimeoutError(RpcError):
    pass


class ListenerClosedError(RpcError):
    pass


class DaemonCrashError(WhatsappConnectorError):
    pass
