"""python-telegram-bot Application wrapper.

Three responsibilities:

1. Build the PTB ``Application`` from the bot token.
2. Discover the bot's identity on startup (numeric id, ``@username``,
   display ``first_name``) via ``Bot.get_me()`` — surfaced to the agent
   in the MCP init instructions (issue #55).  The numeric id doubles as
   the ``<account>`` segment of channel addresses.
3. Register a single message handler that parses each incoming text
   message and POSTs it to aios via :class:`aios_telegram.ingest.IngestClient`.

Long-polling is driven by PTB's own ``Updater``; :func:`run_application`
starts it, blocks until cancelled, then shuts down cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

import structlog
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from .ingest import IngestClient, build_metadata
from .parse import parse_message

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BotIdentity:
    """Bot's own surface on Telegram, captured from ``Bot.get_me()``.

    ``id`` and ``first_name`` are guaranteed by Telegram's schema —
    BotFather sets a ``first_name`` at creation time and the numeric id
    is always present.  ``username`` is optional: a bot can exist without
    one during the initial BotFather configuration window.  The
    renderer in ``prompts.py`` omits ``username`` when unset.
    """

    id: int
    first_name: str
    username: str | None


def build_application(token: str) -> Application:  # type: ignore[type-arg]
    return Application.builder().token(token).build()


async def discover_bot_identity(application: Application) -> BotIdentity:  # type: ignore[type-arg]
    """Call ``getMe`` and return the bot's identity surface.

    Called once on startup after ``application.initialize()``.  PTB's
    ``User`` schema guarantees ``id`` and ``first_name``; a genuinely
    malformed response would raise inside ``get_me`` or violate PTB's
    types, so we trust the fields here.
    """
    me = await application.bot.get_me()
    log.info(
        "telegram.bot.identified",
        bot_id=me.id,
        username=me.username,
        first_name=me.first_name,
    )
    return BotIdentity(
        id=int(me.id),
        first_name=me.first_name,
        username=me.username or None,
    )


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
