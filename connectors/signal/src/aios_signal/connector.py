"""Signal connector ported to the aios-connector SDK.

Replaces the pre-PR3 FastMCP HTTP server + ingest-HTTP-POST architecture
with a single :class:`aios_connector.Connector` subclass communicating
with aios over stdio MCP.

Lifecycle:

* :meth:`setup` opens :class:`SignalDaemon` (which spawns ``signal-cli
  daemon`` and waits for TCP readiness), discovers the bot UUID, and
  loads contacts + groups for display-name resolution.
* :meth:`discover_accounts` returns one account entry — Signal connectors
  are single-bot by design.
* :meth:`serve` drives the inbound pump: drains messages from
  ``daemon.listener``, parses them, and calls :meth:`emit_inbound` for
  each one.  Spool durability + dedup ledger are handled by the SDK.
* :meth:`teardown` closes the daemon (SIGTERM → grace → SIGKILL).
* The two model-facing tools, ``signal_send`` and ``signal_react``,
  use :func:`focal_required` so the focal channel suffix is parsed
  out of ``_meta`` and bound to the ``focal`` kwarg.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import structlog
from aios_connector import Connector, focal_required, make_account, tool

from .addressing import decode_chat_id, encode_chat_id
from .config import Settings
from .daemon import SignalDaemon
from .markdown import convert_markdown_to_signal_styles
from .parse import InboundMessage, build_content_text, parse_envelope
from .prompts import build_instructions

log = structlog.get_logger(__name__)


class SignalConnector(Connector):
    name = "signal"
    version = "0.1.0"

    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self._cfg = cfg
        self._daemon: SignalDaemon | None = None
        self._bot_uuid: str | None = None
        self._contact_names: dict[str, str] = {}

    # ── lifecycle ─────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Open the signal-cli daemon and load contacts + groups.

        signal-cli takes 5+ seconds to come up; the supervisor's bounded
        init handshake (30s) accommodates this.  The SDK doesn't
        finish ``initialize()`` until this method returns, so the
        supervisor stays in ``starting`` and aios won't dispatch tool
        calls against an unready daemon.
        """
        self._daemon = await SignalDaemon(
            phone=self._cfg.phone,
            config_dir=self._cfg.config_dir,
            cli_bin=self._cfg.cli_bin,
            host=self._cfg.daemon_host,
            port=self._cfg.daemon_port,
        ).__aenter__()
        self._bot_uuid = await self._daemon.discover_bot_uuid()
        self._contact_names = await self._daemon.list_contacts()
        groups = await self._daemon.list_groups()
        # Replace instructions with a runtime-built block that captures
        # the bot's identity, contacts, and groups.  The SDK forwards
        # this to aios as ``InitializeResult.instructions``.
        type(self).instructions = build_instructions(
            bot_uuid=self._bot_uuid,
            phone=self._cfg.phone,
            profile_name=self._contact_names.get(self._bot_uuid),
            groups=groups,
            contact_names=self._contact_names,
        )
        log.info(
            "signal.ready",
            bot_uuid=self._bot_uuid,
            phone=self._cfg.phone,
            contacts=len(self._contact_names),
            groups=len(groups),
        )

    async def discover_accounts(self) -> list[dict[str, Any]]:
        assert self._bot_uuid is not None, "setup() must run before discover_accounts()"
        profile_name = self._contact_names.get(self._bot_uuid)
        return [
            make_account(
                id=self._bot_uuid,
                display_name=profile_name or self._cfg.phone,
                metadata={"phone": self._cfg.phone},
            )
        ]

    async def teardown(self) -> None:
        if self._daemon is not None:
            await self._daemon.__aexit__(None, None, None)
            self._daemon = None

    async def serve(self) -> None:
        """Drain inbound envelopes from signal-cli and emit them to aios.

        Falls back on signal-cli's contact store when an envelope's
        ``sourceName`` is empty — Signal's UI resolves names via
        profiles the envelope doesn't carry.
        """
        assert self._daemon is not None, "setup() must run before serve()"
        assert self._bot_uuid is not None
        async for envelope in self._daemon.listener.messages():
            msg = parse_envelope(envelope, bot_account_uuid=self._bot_uuid)
            if msg is None:
                continue
            if msg.sender_name is None:
                resolved = self._contact_names.get(msg.sender_uuid)
                if resolved:
                    msg = replace(msg, sender_name=resolved)
            chat_id = encode_chat_id(msg.raw_chat_id, msg.chat_type)
            content = build_content_text(msg)
            metadata = build_metadata(msg, chat_id, self._bot_uuid)
            sender_payload: dict[str, Any] = {
                "uuid": msg.sender_uuid,
                "display_name": msg.sender_name or msg.sender_uuid,
            }
            await self.emit_inbound(
                account=self._bot_uuid,
                chat_id=chat_id,
                sender=sender_payload,
                content=content,
                metadata=metadata,
            )

    # ── model-facing tools ────────────────────────────────────────────

    @tool()
    @focal_required
    async def signal_send(self, text: str, *, focal: str) -> dict[str, Any]:
        """Send a text message to your focal Signal chat.

        The chat id is taken implicitly from your focal channel —
        aios injects it via the JSON-RPC ``_meta`` field on each call.
        Set focal with the built-in ``switch_channel`` tool.

        Args:
            text: Message body. Markdown is converted to Signal text styles.
        """
        assert self._daemon is not None
        params = _build_send_params(focal, text)
        result = await self._daemon.rpc.call("send", params)
        ts = _extract_timestamp(result)
        return {"sent_at_ms": ts} if ts is not None else {"status": "ok"}

    @tool()
    @focal_required
    async def signal_react(
        self,
        target_author_uuid: str,
        target_timestamp_ms: int,
        emoji: str,
        *,
        focal: str,
    ) -> dict[str, Any]:
        """React to a message in your focal Signal chat with an emoji.

        The chat id is taken implicitly from your focal channel — aios
        injects it via the JSON-RPC ``_meta`` field on each call.

        The target message is identified by ``(target_author_uuid,
        target_timestamp_ms)``.  Every inbound Signal message in your
        conversation starts with a header line like ``[channel=... ·
        from=... · sender_uuid=<uuid> · timestamp_ms=<ms> (<iso>)]``.
        Copy ``sender_uuid`` and the raw ``timestamp_ms`` integer from
        that header; do not construct them yourself.

        Args:
            target_author_uuid: The ``sender_uuid`` from the header of the message
                you're reacting to.
            target_timestamp_ms: The ``timestamp_ms`` integer from the header of the
                message you're reacting to.
            emoji: The reaction emoji.
        """
        assert self._daemon is not None
        params = _build_react_params(focal, target_author_uuid, target_timestamp_ms, emoji)
        await self._daemon.rpc.call("sendReaction", params)
        return {"status": "ok"}


def _build_send_params(chat_id: str, text: str) -> dict[str, Any]:
    """Translate ``(chat_id, text)`` into signal-cli ``send`` params."""
    chat_type, raw_id = decode_chat_id(chat_id)
    stripped, styles = convert_markdown_to_signal_styles(text)
    params: dict[str, Any] = {"message": stripped}
    if styles:
        params["textStyles"] = styles
    if chat_type == "group":
        params["groupId"] = raw_id
    else:
        params["recipient"] = [raw_id]
    return params


def _build_react_params(
    chat_id: str,
    target_author_uuid: str,
    target_timestamp_ms: int,
    emoji: str,
) -> dict[str, Any]:
    """Translate a react request into signal-cli ``sendReaction`` params."""
    chat_type, raw_id = decode_chat_id(chat_id)
    params: dict[str, Any] = {
        "emoji": emoji,
        "targetAuthor": target_author_uuid,
        "targetTimestamp": target_timestamp_ms,
    }
    if chat_type == "group":
        params["groupId"] = raw_id
    else:
        params["recipient"] = [raw_id]
    return params


def build_metadata(msg: InboundMessage, chat_id: str, bot_uuid: str) -> dict[str, Any]:
    """Stamp signal-specific metadata onto an inbound aios event.

    ``channel`` is redundant with what aios stamps server-side, but we
    include it so events are self-describing when read outside aios
    (e.g. ``aios sessions events`` JSON output).  Reply / reaction
    payloads are nested so the model sees them as structured siblings
    of ``content`` rather than embedded prose.
    """
    metadata: dict[str, Any] = {
        "channel": f"signal/{bot_uuid}/{chat_id}",
        "sender_uuid": msg.sender_uuid,
        "timestamp_ms": msg.timestamp_ms,
        "chat_type": msg.chat_type,
    }
    if msg.sender_name is not None:
        metadata["sender_name"] = msg.sender_name
    if msg.chat_name is not None:
        metadata["chat_name"] = msg.chat_name
    if msg.reply is not None:
        metadata["reply_to"] = {
            "author_uuid": msg.reply.author_uuid,
            "timestamp_ms": msg.reply.timestamp_ms,
            "text": msg.reply.text,
        }
    if msg.reaction is not None:
        metadata["reaction"] = {
            "emoji": msg.reaction.emoji,
            "target_author_uuid": msg.reaction.target_author_uuid,
            "target_timestamp_ms": msg.reaction.target_timestamp_ms,
        }
    return metadata


def _extract_timestamp(rpc_result: Any) -> int | None:
    """Pull the send timestamp from signal-cli's ``send`` result, or ``None``.

    signal-cli delivers DM sends with ``{"timestamp": <ms>, ...}``, but
    group sends in 0.14.x return a bare ``null`` even on success — the
    RPC doesn't carry a timestamp.  RPC-level delivery failures raise
    ``RpcError`` in the transport layer, so if we reach this function
    the send *did* happen; a missing timestamp just means we don't
    have an ID to hand back.  Return ``None`` and let the caller
    convey "sent, no ID" to the model.
    """
    if not isinstance(rpc_result, dict):
        return None
    ts = rpc_result.get("timestamp")
    return ts if isinstance(ts, int) else None
