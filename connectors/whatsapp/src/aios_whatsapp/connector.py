"""WhatsApp connector built on the aios-connector-http SDK.

Each connection owns its own ``whatsapp-daemon`` subprocess on an
ephemeral loopback port — whatsmeow's ``Client`` is per-device, so a
daemon per phone keeps lifecycles isolated.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

import structlog
from aios_connector_http import HttpConnector

from .config import Settings
from .daemon import WhatsappDaemon

log = structlog.get_logger(__name__)


@dataclass
class _WhatsappConnectionState:
    phone: str
    daemon: WhatsappDaemon


class WhatsappConnector(HttpConnector):
    connector = "whatsapp"
    state: dict[str, _WhatsappConnectionState]

    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self._cfg = cfg

    async def serve_connection(self, connection_id: str, secrets: dict[str, str]) -> None:
        phone = secrets.get("phone")
        if not phone:
            raise RuntimeError(
                f"whatsapp connection {connection_id!r} requires a 'phone' entry in its secrets"
            )

        store_dir = self._cfg.data_dir / phone
        port = _pick_free_port(self._cfg.daemon_host)
        async with WhatsappDaemon(
            daemon_bin=self._cfg.daemon_bin,
            host=self._cfg.daemon_host,
            port=port,
            store_dir=store_dir,
        ) as daemon:
            self.state[connection_id] = _WhatsappConnectionState(phone=phone, daemon=daemon)
            log.info(
                "whatsapp.connection.ready",
                connection_id=connection_id,
                phone=phone,
                port=port,
            )
            await self._drain_notifications(connection_id, daemon)

    async def _drain_notifications(self, connection_id: str, daemon: WhatsappDaemon) -> None:
        # No method dispatch yet — log unhandled to surface wiring bugs.
        async for method, params in daemon.listener.notifications():
            log.warning(
                "whatsapp.notification.unhandled",
                connection_id=connection_id,
                method=method,
                params=params,
            )


def _pick_free_port(host: str) -> int:
    """Bind to an OS-assigned loopback port and release it.

    Small race window between release and the daemon's bind; the daemon
    raises on EADDRINUSE so the operator gets a loud failure rather
    than a silently-broken connection.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        port: int = s.getsockname()[1]
    return port
