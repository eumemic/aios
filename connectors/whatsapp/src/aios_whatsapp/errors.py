"""Exception hierarchy for aios-whatsapp."""

from __future__ import annotations

from typing import Any


class WhatsappConnectorError(Exception):
    """Base class for all aios-whatsapp errors."""


class RpcError(WhatsappConnectorError):
    """JSON-RPC error from the WhatsApp daemon.

    ``code`` / ``data`` carry the structured fields when present —
    pairing-expired puts diagnostic data in ``data`` and would be lost
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


class DaemonCrashError(WhatsappConnectorError):
    pass
