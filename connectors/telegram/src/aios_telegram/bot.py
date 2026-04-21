"""python-telegram-bot Application wrapper.

Three responsibilities:

1. Build the PTB ``Application`` from the bot token.
2. Discover the bot's numeric id on startup (the ``<account>`` segment of
   channel addresses) via ``Bot.get_me()``.
3. Register a single message handler that parses each incoming text
   message and POSTs it to aios via :class:`aios_telegram.ingest.IngestClient`.

Long-polling is driven by PTB's own ``Updater``; :func:`run_application`
starts it, blocks until cancelled, then shuts down cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib

import structlog
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from .errors import BotIdentityError
from .ingest import IngestClient, build_metadata
from .parse import parse_message

log = structlog.get_logger(__name__)


def build_application(token: str) -> Application:  # type: ignore[type-arg]
    return Application.builder().token(token).build()


async def discover_bot_id(application: Application) -> int:  # type: ignore[type-arg]
    """Call ``getMe`` and return the bot's numeric user id.

    Raises :class:`BotIdentityError` if Telegram's response is unusable.
    Called once on startup after ``application.initialize()``.
    """
    me = await application.bot.get_me()
    if not me.id:
        raise BotIdentityError("getMe returned no id")
    log.info("telegram.bot.identified", bot_id=me.id, username=me.username)
    return int(me.id)


def install_handler(
    application: Application,  # type: ignore[type-arg]
    *,
    bot_id: int,
    ingest: IngestClient,
) -> None:
    """Wire the parse → build_metadata → post_message pipeline onto PTB.

    Filter: plain text messages only (no commands — commands look like text
    to the agent). Non-text payloads and edits are dropped by the filter;
    self / bot-to-bot / empty-text are dropped by :func:`parse_message`.
    """

    async def on_message(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None:
            return
        parsed = parse_message(message, bot_id=bot_id)
        if parsed is None:
            return
        await ingest.post_message(
            path=str(parsed.chat_id),
            content=parsed.text,
            metadata=build_metadata(parsed, bot_id),
        )

    async def on_error(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("telegram.handler.error", error=str(context.error))

    application.add_handler(MessageHandler(filters.TEXT, on_message))
    application.add_error_handler(on_error)


async def run_application(application: Application) -> None:  # type: ignore[type-arg]
    """Run PTB's long-polling loop until cancelled.

    Assumes ``application.initialize()`` has already been awaited by the
    caller — :func:`app.run` does so eagerly to allow ``getMe`` before
    polling starts. Lifecycle here: start → start_polling → (wait
    forever) → stop_polling → stop → shutdown. Cancellation from the
    enclosing TaskGroup triggers the shutdown half.
    """
    await application.start()
    assert application.updater is not None
    await application.updater.start_polling()
    try:
        await asyncio.Event().wait()
    finally:
        with contextlib.suppress(Exception):
            await application.updater.stop()
        with contextlib.suppress(Exception):
            await application.stop()
        with contextlib.suppress(Exception):
            await application.shutdown()
