"""Telegram connector ported to the aios-connector SDK.

Replaces the pre-PR3 FastMCP HTTP server + ingest-HTTP-POST architecture
with a single :class:`aios_connector.Connector` subclass communicating
with aios over stdio MCP.

Per-account paradigm: each Telegram bot token is a distinct platform
identity (PTB's :class:`Application` is bound 1:1 to a token), so this
connector is single-bot by design.  Operators deploy multiple bots by
listing multiple instances under the same connector type (e.g.
``connectors_enabled=telegram:support,telegram:alerts`` with
``AIOS_TELEGRAM_SUPPORT_BOT_TOKEN`` / ``AIOS_TELEGRAM_ALERTS_BOT_TOKEN``).
The supervisor spawns one subprocess per instance; each runs one PTB
``Application`` and reports a single account.

Lifecycle:

* :meth:`setup` initializes the python-telegram-bot ``Application`` and
  discovers the bot's identity via ``Bot.get_me()`` (numeric id +
  ``first_name`` + optional ``@username``).
* :meth:`discover_accounts` returns the one bot account.
* :meth:`serve` starts PTB's long-polling loop and routes inbound
  messages to :meth:`emit_inbound`.  PTB runs its own background tasks;
  we just install a handler that funnels into :meth:`emit_inbound`.
* :meth:`teardown` stops polling and shuts down the application cleanly.
* The single model-facing tool ``telegram_send`` uses :func:`focal_required`
  with only ``chat_id`` in its signature — the SDK injects nothing else,
  so connector code stays focused on the per-bot logic.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import structlog
from aios_connector import Connector, focal_required, make_account, tool
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from .config import Settings
from .parse import InboundMessage, parse_message
from .prompts import build_instructions

log = structlog.get_logger(__name__)


class TelegramConnector(Connector):
    name = "telegram"
    version = "0.1.0"

    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self._cfg = cfg
        self._application: Application | None = None  # type: ignore[type-arg]
        self._bot_id: int | None = None
        self._first_name: str | None = None
        self._username: str | None = None
        self._inbound_queue: asyncio.Queue[InboundMessage] | None = None

    # ── lifecycle ─────────────────────────────────────────────────────

    async def setup(self) -> None:
        application = Application.builder().token(self._cfg.bot_token).build()
        await application.initialize()
        try:
            me = await application.bot.get_me()
        except BaseException:
            await application.shutdown()
            raise

        self._application = application
        self._bot_id = int(me.id)
        self._first_name = me.first_name
        self._username = me.username or None
        self._inbound_queue = asyncio.Queue()

        # Set on instance, not class, so a second connector in tests
        # doesn't see leakage from the first.
        self.instructions = build_instructions(
            bot_id=self._bot_id,
            first_name=self._first_name,
            username=self._username,
        )

        # Install message handler that pushes parsed inbound onto the
        # queue.  PTB uses its own background tasks for long-polling;
        # we keep handler bodies tiny so PTB's worker doesn't block on
        # our spool write.  References go through ``self`` so a future
        # ``setup`` re-run wouldn't leave the closure pointing at a
        # stale queue.
        async def on_message(update: Any, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
            message = update.message
            if message is None:
                return
            assert self._bot_id is not None
            assert self._inbound_queue is not None
            parsed = parse_message(message, bot_id=self._bot_id)
            if parsed is None:
                return
            await self._inbound_queue.put(parsed)

        async def on_error(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
            log.error("telegram.handler.error", error=str(context.error))

        application.add_handler(MessageHandler(filters.TEXT, on_message))
        application.add_error_handler(on_error)
        log.info(
            "telegram.bot.identified",
            bot_id=self._bot_id,
            username=self._username,
            first_name=self._first_name,
        )

    async def discover_accounts(self) -> list[dict[str, Any]]:
        assert self._bot_id is not None and self._first_name is not None
        metadata: dict[str, Any] = {"first_name": self._first_name}
        if self._username:
            metadata["username"] = self._username
        return [
            make_account(
                id=str(self._bot_id),
                display_name=self._first_name,
                metadata=metadata,
            )
        ]

    async def teardown(self) -> None:
        if self._application is None:
            return
        with contextlib.suppress(Exception):
            if self._application.updater is not None:
                await self._application.updater.stop()
        with contextlib.suppress(Exception):
            await self._application.stop()
        with contextlib.suppress(Exception):
            await self._application.shutdown()
        self._application = None

    async def serve(self) -> None:
        """Start PTB long-polling and forward inbound messages to aios.

        Two coroutines run concurrently: PTB's polling (which fills the
        queue via the registered handler) and our inbound drainer
        (which pulls from the queue and calls :meth:`emit_inbound`).
        Cancelling either propagates and ``teardown`` runs in
        :meth:`Connector.run`'s finally.
        """
        assert self._application is not None
        assert self._inbound_queue is not None
        assert self._bot_id is not None

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._run_polling(), name="telegram-polling")
            tg.create_task(self._drain_queue(), name="telegram-drain")

    async def _run_polling(self) -> None:
        assert self._application is not None
        await self._application.start()
        assert self._application.updater is not None
        await self._application.updater.start_polling()
        await asyncio.Event().wait()

    async def _drain_queue(self) -> None:
        assert self._inbound_queue is not None
        assert self._bot_id is not None
        while True:
            msg = await self._inbound_queue.get()
            sender_payload: dict[str, Any] = {
                "id": msg.sender_id,
                "display_name": msg.sender_name or str(msg.sender_id),
            }
            metadata = build_metadata(msg, self._bot_id)
            # Telegram message ``date`` is unix-seconds; render as
            # ISO-8601 UTC so aios's supervisor sees the same string
            # shape as other connectors (signal etc.).
            timestamp_iso = (
                datetime.fromtimestamp(msg.timestamp_ms / 1000, tz=UTC).isoformat()
                if msg.timestamp_ms
                else None
            )
            await self.emit_inbound(
                account=str(self._bot_id),
                chat_id=str(msg.chat_id),
                sender=sender_payload,
                content=msg.text,
                metadata=metadata,
                timestamp=timestamp_iso,
            )

    # ── model-facing tools ────────────────────────────────────────────

    @tool()
    @focal_required
    async def telegram_send(self, text: str, *, chat_id: str) -> dict[str, Any]:
        """Send a text message to your focal Telegram chat.

        The chat id is taken implicitly from your focal channel — aios
        injects it via the JSON-RPC ``_meta`` field on each call.  Set
        focal with the built-in ``switch_channel`` tool.

        Args:
            text: Message body. Plain text only — markdown is not rendered.
        """
        assert self._application is not None
        try:
            chat_id_int = int(chat_id)
        except ValueError as e:
            raise ValueError(f"telegram chat_id must be an integer; got {chat_id!r}") from e
        sent = await self._application.bot.send_message(chat_id=chat_id_int, text=text)
        return {"message_id": sent.message_id}


def build_metadata(msg: InboundMessage, bot_id: int) -> dict[str, Any]:
    """Stamp telegram-specific metadata onto an inbound aios event.

    ``channel`` is redundant with what aios stamps server-side, but we
    include it so events are self-describing when read outside aios
    (e.g. ``aios sessions events`` JSON output).  Reply-payload is
    nested so the model sees it as a structured sibling of ``content``
    rather than embedded prose.
    """
    metadata: dict[str, Any] = {
        "channel": f"telegram/{bot_id}/{msg.chat_id}",
        "chat_type": msg.chat_kind,
        "sender_id": msg.sender_id,
        "message_id": msg.message_id,
        "timestamp_ms": msg.timestamp_ms,
    }
    if msg.sender_name is not None:
        metadata["sender_name"] = msg.sender_name
    if msg.chat_name is not None:
        metadata["chat_name"] = msg.chat_name
    if msg.reply is not None:
        metadata["reply_to"] = {
            "message_id": msg.reply.message_id,
            "text": msg.reply.text,
        }
    return metadata
