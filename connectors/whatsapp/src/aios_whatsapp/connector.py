"""WhatsApp connector built on the aios-connector-http SDK.

Each connection owns its own ``whatsapp-daemon`` subprocess on an
ephemeral loopback port — whatsmeow's ``Client`` is per-device, so a
daemon per phone keeps lifecycles isolated.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any

import structlog
from aios_connector_http import HttpConnector, iso_from_ms, tool

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
        metadata: dict[str, Any] = {
            "chat_type": msg.chat_type,
            "sender_jid": msg.sender_jid,
            "message_id": msg.message_id,
        }
        if msg.chat_name is not None:
            metadata["chat_name"] = msg.chat_name
        await self.emit_inbound(
            connection_id=connection_id,
            event_id=f"whatsapp-{msg.sender_jid}-{msg.message_id}",
            chat_id=msg.chat_jid,
            sender={"jid": msg.sender_jid, "display_name": msg.sender_name},
            content=msg.text,
            metadata=metadata,
            timestamp=iso_from_ms(msg.timestamp_ms),
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
        state = self.state[connection_id]
        result = await state.daemon.rpc.call(
            "sendMessage",
            {"jid": chat_id, "text": text},
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"sendMessage returned non-dict: {result!r}")
        return result


def _pick_free_port(host: str) -> int:
    """Bind to an OS-assigned loopback port and immediately release it.

    Tiny race window between the release and the daemon's subsequent
    bind; if a third party grabs the port in between, the daemon's
    spawn fails with EADDRINUSE and ``_wait_for_tcp`` surfaces that as
    :class:`DaemonCrashError` — the loud failure the operator wants
    rather than a silently-broken connection.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        port: int = s.getsockname()[1]
    return port
