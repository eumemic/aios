"""WhatsApp connector built on the aios-connector-http SDK.

Each connection owns its own ``whatsapp-daemon`` subprocess on an
ephemeral loopback port — whatsmeow's ``Client`` is per-device, so a
daemon per phone keeps lifecycles isolated.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from aios_connector_http import HttpConnector, tool

from .addressing import is_valid_chat_id
from .config import Settings
from .daemon import WhatsappDaemon
from .parse import parse_message

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
            await self._dispatch_notifications(connection_id, daemon)

    async def _dispatch_notifications(self, connection_id: str, daemon: WhatsappDaemon) -> None:
        async for method, params in daemon.listener.notifications():
            if method == "message":
                await self._handle_inbound_message(connection_id, params)
            else:
                log.warning(
                    "whatsapp.notification.unhandled",
                    connection_id=connection_id,
                    method=method,
                    params=params,
                )

    async def _handle_inbound_message(self, connection_id: str, params: dict[str, Any]) -> None:
        msg = parse_message(params)
        if msg is None:
            return
        await self.emit_inbound(
            connection_id=connection_id,
            event_id=f"whatsapp-{msg.sender_jid}-{msg.message_id}",
            chat_id=msg.chat_jid,
            sender={"jid": msg.sender_jid, "display_name": msg.sender_name},
            content=msg.text,
            metadata={
                "chat_type": msg.chat_type,
                "chat_jid": msg.chat_jid,
                "chat_name": msg.chat_name,
                "sender_jid": msg.sender_jid,
                "message_id": msg.message_id,
                "timestamp_ms": msg.timestamp_ms,
            },
            timestamp=_iso(msg.timestamp_ms),
        )

    # ── tools ──────────────────────────────────────────────────────────

    @tool()
    async def whatsapp_send(
        self,
        text: str,
        *,
        connection_id: str,
        chat_id: str,
    ) -> dict[str, Any]:
        """Send a text message to your focal WhatsApp chat.

        Args:
            text: The message body.

        Returns:
            ``{"message_id": "...", "timestamp_ms": ...}`` from the
            daemon's whatsmeow SendMessage round-trip.
        """
        if not is_valid_chat_id(chat_id):
            raise ValueError(f"invalid WhatsApp chat_id: {chat_id!r}")
        state = self.state[connection_id]
        result = await state.daemon.rpc.call(
            "sendMessage",
            {"jid": chat_id, "text": text},
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"sendMessage returned non-dict: {result!r}")
        return result


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


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()
