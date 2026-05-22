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
from .management import WhatsappManagementMixin, normalize_phone
from .parse import parse_message

log = structlog.get_logger(__name__)


@dataclass
class _WhatsappConnectionState:
    phone: str
    daemon: WhatsappDaemon


class WhatsappConnector(WhatsappManagementMixin, HttpConnector):
    connector = "whatsapp"
    state: dict[str, _WhatsappConnectionState]

    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self._cfg = cfg

    async def serve_connection(self, connection_id: str, secrets: dict[str, str]) -> None:
        phone_raw = secrets.get("phone")
        if not phone_raw:
            raise RuntimeError(
                f"whatsapp connection {connection_id!r} requires a 'phone' entry in its secrets"
            )
        # Normalize at this boundary so _state_for_phone's lookup
        # works regardless of how the operator formatted the phone
        # at connection-create time vs management-call time.
        phone = normalize_phone(phone_raw)

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

    @tool()
    async def whatsapp_react(
        self,
        message_id: str,
        reaction: str,
        *,
        connection_id: str,
    ) -> dict[str, Any]:
        """React to a previously-seen WhatsApp message.

        Args:
            message_id: The id of the message to react to.  Take this
                from the ``message_id`` field of the inbound metadata
                you're targeting (visible in the channel headers
                rendered into your context).
            reaction: The reaction emoji.  Pass an empty string to
                clear any prior reaction you placed on the message.

        Returns:
            ``{"message_id": "...", "timestamp_ms": ...}`` of the
            reaction send itself.  Raises if the daemon has never seen
            the target message — typically because it predates the bot
            joining the chat or is older than the daemon's local index.
        """
        state = self.state[connection_id]
        result = await state.daemon.rpc.call(
            "sendReaction",
            {"message_id": message_id, "reaction": reaction},
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"sendReaction returned non-dict: {result!r}")
        return result

    @tool()
    async def whatsapp_edit_message(
        self,
        message_id: str,
        text: str,
        *,
        connection_id: str,
    ) -> dict[str, Any]:
        """Edit one of your own previously-sent WhatsApp messages.

        WhatsApp only allows editing messages you sent, and only within
        ~15 minutes of the original send; the daemon refuses outside
        either window with a structured error you can read in the
        failure path.

        Args:
            message_id: The id of your prior outbound message.
            text: The replacement body.

        Returns:
            ``{"message_id": "...", "timestamp_ms": ...}`` of the edit
            envelope (a separate, distinct id from the original).
        """
        state = self.state[connection_id]
        result = await state.daemon.rpc.call(
            "editMessage",
            {"message_id": message_id, "text": text},
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"editMessage returned non-dict: {result!r}")
        return result

    @tool()
    async def whatsapp_delete_message(
        self,
        message_id: str,
        *,
        connection_id: str,
    ) -> dict[str, Any]:
        """Delete one of your own previously-sent WhatsApp messages
        (the \"delete for everyone\" action).

        Like editing, deletion is only valid on your own outbounds;
        the daemon enforces this rather than waiting for WhatsApp's
        server to reject with an opaque error.

        Args:
            message_id: The id of your prior outbound message.

        Returns:
            ``{"message_id": "...", "timestamp_ms": ...}`` of the
            revoke envelope.
        """
        state = self.state[connection_id]
        result = await state.daemon.rpc.call(
            "deleteMessage",
            {"message_id": message_id},
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"deleteMessage returned non-dict: {result!r}")
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
