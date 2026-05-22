"""WhatsApp connector built on the aios-connector-http SDK.

Each connection owns its own ``whatsapp-daemon`` subprocess on an
ephemeral loopback port — whatsmeow's ``Client`` is per-device, so a
daemon per phone keeps lifecycles isolated.
"""

from __future__ import annotations

import asyncio
import mimetypes
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from aios_connector_http import HttpConnector, SandboxPath, iso_from_ms, tool

from .config import Settings
from .daemon import WhatsappDaemon
from .format import markdown_to_whatsapp
from .management import WhatsappManagementMixin, normalize_phone
from .parse import InboundMessage, parse_message

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
        if msg.sticker_emoji is not None:
            metadata["sticker_emoji"] = msg.sticker_emoji
        if msg.reaction is not None:
            metadata["reaction"] = {
                "emoji": msg.reaction.emoji,
                "target_message_id": msg.reaction.target_message_id,
            }
        if msg.edit_target_message_id is not None:
            metadata["edited"] = True
            metadata["edit_target_message_id"] = msg.edit_target_message_id
        if msg.revoke_target_message_id is not None:
            metadata["revoked"] = True
            metadata["revoke_target_message_id"] = msg.revoke_target_message_id
        attachment_tuples = await self._read_attachments(msg) if msg.attachments else None
        if msg.attachments and attachment_tuples is None:
            # Daemon declared attachments but every Path.read_bytes
            # raised — the model would otherwise see an empty-content
            # event with no signal that bytes were lost.  Stamp a
            # diagnostic so the model can apologise to the user
            # rather than ignore them.
            metadata["attachments_unreadable"] = len(msg.attachments)
        await self.emit_inbound(
            connection_id=connection_id,
            event_id=f"whatsapp-{msg.sender_jid}-{msg.message_id}",
            chat_id=msg.chat_jid,
            sender={"jid": msg.sender_jid, "display_name": msg.sender_name},
            content=msg.text,
            attachments=attachment_tuples,
            metadata=metadata,
            timestamp=iso_from_ms(msg.timestamp_ms),
        )

    async def _read_attachments(self, msg: InboundMessage) -> list[tuple[str, bytes, str]] | None:
        """Pull each attachment's bytes off the event loop so a multi-MiB
        photo doesn't stall the inbound dispatcher.

        A read failure on any one attachment drops THAT entry but
        keeps the others — preferable to discarding the whole message
        when the daemon wrote two files and one got truncated.
        Returns None if nothing readable survives (so emit_inbound
        sees no ``attachments`` kwarg rather than an empty list).
        """
        results: list[tuple[str, bytes, str]] = []
        for att in msg.attachments:
            try:
                data = await asyncio.to_thread(Path(att.host_path).read_bytes)
            except OSError as err:
                log.warning(
                    "whatsapp.inbound.attachment_read_failed",
                    host_path=att.host_path,
                    error=str(err),
                )
                continue
            results.append((att.filename, data, att.content_type))
        return results or None

    # ── tools ──────────────────────────────────────────────────────────

    @tool()
    async def whatsapp_send(
        self,
        text: str,
        attachments: list[SandboxPath] | None = None,
        *,
        connection_id: str,
        chat_id: str,
    ) -> dict[str, Any]:
        """Send a message to your focal WhatsApp chat.

        Args:
            text: The message body.  When ``attachments`` is set, this
                becomes the caption on the FIRST attachment only;
                subsequent attachments arrive caption-less (WhatsApp
                has no media-group equivalent — each attachment is its
                own message).  Pass an empty string to send the
                attachment(s) without any caption.
            attachments: Optional in-sandbox file paths to attach.
                The SDK resolves each entry to a host path before
                this method runs.  Mimetype is derived from the file
                extension via Python's ``mimetypes`` module; unknown
                types fall through to a generic document send.

        Returns:
            ``{"message_id": "...", "timestamp_ms": ...}`` of the
            FIRST sent message — for multi-attachment sends, that's
            the caption-bearer.  Subsequent attachment ids land in
            the daemon's message store too and remain
            react/edit/delete-targetable individually.
        """
        state = self.state[connection_id]
        params: dict[str, Any] = {"jid": chat_id, "text": markdown_to_whatsapp(text)}
        if attachments:
            params["attachments"] = [_attachment_params(p) for p in attachments]
        result = await state.daemon.rpc.call("sendMessage", params)
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
            {"message_id": message_id, "text": markdown_to_whatsapp(text)},
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


def _attachment_params(host_path: Path) -> dict[str, str]:
    """Build the daemon RPC's attachment dict from a SandboxPath-resolved
    host path.

    Mimetype is derived from the file extension via Python's
    ``mimetypes`` module; unknown types fall through to
    ``application/octet-stream`` and the daemon classifies the
    attachment as a document (whatsmeow's catch-all kind).  Filename
    appears on the WhatsApp wire only for documents, but the daemon
    carries it for every kind so inbound clients can see a meaningful
    label.
    """
    mime, _ = mimetypes.guess_type(host_path.name)
    return {
        "path": str(host_path),
        "mimetype": mime or "application/octet-stream",
        "filename": host_path.name,
    }


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
