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
  messages, edits, and reactions to :meth:`emit_inbound`.  PTB runs
  its own background tasks; we just install handlers that funnel into
  an internal queue.
* :meth:`teardown` stops polling and shuts down the application cleanly.

Tools exposed to the model: ``telegram_send``, ``telegram_typing``,
``telegram_edit_message``, ``telegram_delete_message``, ``telegram_react``.
All use :func:`focal_required` with ``chat_id`` so the focal channel
selects the target chat without the model having to thread it through.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from aios_connector import (
    Attachment as SDKAttachment,
)
from aios_connector import (
    AttachmentError,
    Connector,
    SandboxPath,
    focal_required,
    make_account,
    tool,
)
from telegram import (
    InputMediaAudio,
    InputMediaDocument,
    InputMediaPhoto,
    InputMediaVideo,
    ReactionTypeEmoji,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    MessageReactionHandler,
    filters,
)

from .config import Settings
from .format import markdown_to_telegram_html
from .parse import (
    Attachment,
    InboundMessage,
    InboundReaction,
    parse_message,
    parse_reaction,
)
from .prompts import build_instructions

log = structlog.get_logger(__name__)


_ALLOWED_UPDATES: list[str] = [
    Update.MESSAGE,
    Update.EDITED_MESSAGE,
    Update.MESSAGE_REACTION,
]

_PARSE_MODE_TO_PTB: dict[str, str | None] = {"plain": None, "html": "HTML"}


class TelegramConnector(Connector):
    name = "telegram"
    version = "0.2.0"

    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self._cfg = cfg
        self._application: Application | None = None  # type: ignore[type-arg]
        self._bot_id: int | None = None
        self._first_name: str | None = None
        self._username: str | None = None
        self._inbound_queue: asyncio.Queue[InboundMessage | InboundReaction] | None = None

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

        # Install handlers that push parsed inbound onto the queue.
        # Handler bodies are tiny so PTB's worker doesn't block on our
        # spool write — references go through ``self`` so a future
        # ``setup`` re-run won't leave the closure pointing at a stale
        # queue.
        async def on_message(update: Any, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
            message = update.message or update.edited_message
            if message is None:
                return
            assert self._bot_id is not None
            assert self._inbound_queue is not None
            parsed = parse_message(message, bot_id=self._bot_id)
            if parsed is None:
                return
            await self._inbound_queue.put(parsed)

        async def on_reaction(update: Any, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
            reaction = update.message_reaction
            if reaction is None:
                return
            assert self._bot_id is not None
            assert self._inbound_queue is not None
            parsed = parse_reaction(reaction, bot_id=self._bot_id)
            if parsed is None:
                return
            await self._inbound_queue.put(parsed)

        async def on_error(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
            log.error("telegram.handler.error", error=str(context.error))

        # filters.UpdateType.MESSAGES matches both new and edited messages;
        # parse_message handles channel/bot filtering with full message context.
        application.add_handler(MessageHandler(filters.UpdateType.MESSAGES, on_message))
        application.add_handler(MessageReactionHandler(on_reaction))
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
        # ``allowed_updates`` is opt-in — Telegram only delivers update
        # types we explicitly subscribe to.  Without this, edits and
        # reactions never reach the bot regardless of which handlers we
        # register locally.
        await self._application.updater.start_polling(allowed_updates=_ALLOWED_UPDATES)
        await asyncio.Event().wait()

    async def _drain_queue(self) -> None:
        assert self._inbound_queue is not None
        assert self._bot_id is not None
        assert self._application is not None
        while True:
            item = await self._inbound_queue.get()
            if isinstance(item, InboundReaction):
                await self._emit_reaction(item)
            else:
                await self._emit_message(item)

    async def _emit_message(self, msg: InboundMessage) -> None:
        assert self._bot_id is not None
        sender_payload: dict[str, Any] = {
            "id": msg.sender_id,
            "display_name": msg.sender_name or str(msg.sender_id),
        }
        metadata = build_metadata(msg, self._bot_id)
        # Telegram's ``message.date`` is unix-seconds; the parser stamps
        # it as ``timestamp_ms``.  Render that as ISO-8601 UTC so aios's
        # supervisor sees the same string shape as other connectors.
        timestamp_iso = (
            datetime.fromtimestamp(msg.timestamp_ms / 1000, tz=UTC).isoformat()
            if msg.timestamp_ms
            else None
        )
        sdk_attachments = await self._download_attachments(msg.attachments)
        await self.emit_inbound(
            account=str(self._bot_id),
            chat_id=str(msg.chat_id),
            sender=sender_payload,
            content=msg.text,
            attachments=sdk_attachments or None,
            metadata=metadata,
            timestamp=timestamp_iso,
        )

    async def _emit_reaction(self, reaction: InboundReaction) -> None:
        assert self._bot_id is not None
        sender_payload: dict[str, Any] = {
            "id": reaction.sender_id,
            "display_name": reaction.sender_name or str(reaction.sender_id),
        }
        metadata: dict[str, Any] = {
            "channel": f"telegram/{self._bot_id}/{reaction.chat_id}",
            "chat_type": reaction.chat_kind,
            "sender_id": reaction.sender_id,
            "timestamp_ms": reaction.timestamp_ms,
            "reaction": {
                "target_message_id": reaction.target_message_id,
                "old_emojis": list(reaction.old_emojis),
                "new_emojis": list(reaction.new_emojis),
            },
        }
        if reaction.sender_name is not None:
            metadata["sender_name"] = reaction.sender_name
        if reaction.chat_name is not None:
            metadata["chat_name"] = reaction.chat_name
        timestamp_iso = (
            datetime.fromtimestamp(reaction.timestamp_ms / 1000, tz=UTC).isoformat()
            if reaction.timestamp_ms
            else None
        )
        await self.emit_inbound(
            account=str(self._bot_id),
            chat_id=str(reaction.chat_id),
            sender=sender_payload,
            content="",
            metadata=metadata,
            timestamp=timestamp_iso,
        )

    async def _download_attachments(
        self, attachments: tuple[Attachment, ...]
    ) -> list[SDKAttachment]:
        """Download each attachment in parallel, log+skip the rejects."""
        host_paths = await asyncio.gather(*(self._download_one(a) for a in attachments))
        out: list[SDKAttachment] = []
        for att, host_path in zip(attachments, host_paths, strict=True):
            if host_path is None:
                continue
            candidate = SDKAttachment(
                host_path=str(host_path),
                filename=att.filename,
                content_type=att.content_type,
            )
            try:
                candidate.as_params()
            except AttachmentError as err:
                log.warning(
                    "telegram.inbound.attachment_rejected",
                    file_id=att.file_id,
                    filename=att.filename,
                    error=str(err),
                )
                continue
            out.append(candidate)
        return out

    async def _download_one(self, att: Attachment) -> Path | None:
        assert self._application is not None
        try:
            file = await self._application.bot.get_file(att.file_id)
        except TelegramError as err:
            log.warning(
                "telegram.inbound.get_file_failed",
                file_id=att.file_id,
                error=str(err),
            )
            return None
        # Close the handle immediately so PTB owns the write side.
        with tempfile.NamedTemporaryFile(
            prefix="aios-telegram-", suffix=Path(att.filename).suffix, delete=False
        ) as tmp:
            target = Path(tmp.name)
        try:
            await file.download_to_drive(custom_path=target)
        except (TelegramError, OSError) as err:
            log.warning(
                "telegram.inbound.download_failed",
                file_id=att.file_id,
                target=str(target),
                error=str(err),
            )
            await asyncio.to_thread(target.unlink, missing_ok=True)
            return None
        return target

    # ── model-facing tools ────────────────────────────────────────────

    @tool()
    @focal_required
    async def telegram_send(
        self,
        text: str,
        attachments: list[SandboxPath] | None = None,
        parse_mode: Literal["plain", "html"] = "plain",
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """Send a Telegram message to your focal chat, optionally with attachments.

        The chat id is taken implicitly from your focal channel — aios
        injects it via the JSON-RPC ``_meta`` field on each call.  Set
        focal with the built-in ``switch_channel`` tool.

        Args:
            text: Message body.  Becomes the caption when attachments are
                present.  See ``parse_mode``.
            attachments: Optional in-sandbox file paths.  Type is inferred
                from extension (``.jpg``/``.png``/``.gif``/``.webp`` →
                photo, ``.mp4``/``.mov`` → video, ``.ogg`` → voice,
                ``.mp3``/``.m4a``/``.wav`` → audio, anything else →
                document).  Single attachment uses ``send_photo`` /
                ``send_voice`` / etc.; multiple attachments use
                ``send_media_group`` with caption attached to the first
                item only (per Telegram API).
            parse_mode: ``"plain"`` (default) sends the text as-is —
                literal characters, no formatting.  ``"html"`` runs
                ``text`` through a Markdown→Telegram-HTML converter so
                ``**bold**``, ``*italic*``, ``[label](url)``, fenced
                code, ``> quotes``, and ``||spoilers||`` render with
                Telegram's native styling.
        """
        assert self._application is not None
        chat_id_int = _coerce_chat_id(chat_id)

        body, ptb_parse_mode = _prepare_text(text, parse_mode)
        host_paths: list[Path] = list(attachments or [])
        bot = self._application.bot

        if not host_paths:
            sent = await bot.send_message(
                chat_id=chat_id_int,
                text=body,
                parse_mode=ptb_parse_mode,
            )
            return {"message_id": sent.message_id}

        if len(host_paths) == 1:
            single = await _send_single_media(
                bot,
                chat_id=chat_id_int,
                host_path=host_paths[0],
                caption=body or None,
                parse_mode=ptb_parse_mode,
            )
            return {"message_id": single.message_id}

        sent_group = await bot.send_media_group(
            chat_id=chat_id_int,
            media=_build_media_group(host_paths, caption=body or None, parse_mode=ptb_parse_mode),
        )
        return {"message_ids": [m.message_id for m in sent_group]}

    @tool()
    @focal_required
    async def telegram_typing(
        self,
        action: Literal[
            "typing", "upload_photo", "record_voice", "upload_voice", "upload_document"
        ] = "typing",
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """Show a chat-action bubble (e.g. "typing…") in your focal chat.

        Telegram displays the action for up to 5 seconds or until the
        next message arrives.  Call this *before* doing slow work (a
        long tool run, an LLM hop) so users see you're alive; you don't
        need to keep re-calling it on a timer for typical replies.

        Args:
            action: Which bubble to show.  ``"typing"`` is the default
                and right for plain replies; ``"upload_photo"`` /
                ``"record_voice"`` / ``"upload_voice"`` /
                ``"upload_document"`` make sense before sending media.
        """
        assert self._application is not None
        chat_id_int = _coerce_chat_id(chat_id)
        await self._application.bot.send_chat_action(chat_id=chat_id_int, action=ChatAction(action))
        return {"status": "ok"}

    @tool()
    @focal_required
    async def telegram_edit_message(
        self,
        message_id: int,
        text: str,
        parse_mode: Literal["plain", "html"] = "plain",
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """Replace the text of a message you sent earlier in your focal chat.

        Useful for streaming-style updates ("I'm thinking…" → final
        answer) or correcting a typo without spamming a follow-up.
        Telegram only lets you edit your own messages and only within
        48 hours of sending.

        Args:
            message_id: The id of the message to edit.  Returned by
                ``telegram_send`` as ``message_id``.
            text: The new body.  See ``parse_mode``.
            parse_mode: ``"plain"`` sends literal text; ``"html"`` runs
                ``text`` through the same Markdown→Telegram-HTML
                converter as ``telegram_send``.
        """
        assert self._application is not None
        chat_id_int = _coerce_chat_id(chat_id)
        body, ptb_parse_mode = _prepare_text(text, parse_mode)
        edited = await self._application.bot.edit_message_text(
            chat_id=chat_id_int,
            message_id=message_id,
            text=body,
            parse_mode=ptb_parse_mode,
        )
        # ``edit_message_text`` returns ``True`` when the message is an
        # inline-bot message we don't own, otherwise the edited Message.
        if isinstance(edited, bool):
            return {"status": "ok"}
        return {"message_id": edited.message_id}

    @tool()
    @focal_required
    async def telegram_delete_message(
        self,
        message_id: int,
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """Delete a message in your focal chat by id.

        You can delete your own messages within 48 hours.  In groups,
        admins can delete others' messages — but bots only get that
        permission when explicitly granted "Delete Messages."

        Args:
            message_id: The id of the message to delete.
        """
        assert self._application is not None
        chat_id_int = _coerce_chat_id(chat_id)
        await self._application.bot.delete_message(chat_id=chat_id_int, message_id=message_id)
        return {"status": "ok"}

    @tool()
    @focal_required
    async def telegram_react(
        self,
        message_id: int,
        emoji: str | None,
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """React to a message in your focal chat with an emoji, or clear your reaction.

        Bots can set at most one reaction per message.  Pass
        ``emoji=None`` to clear the bot's existing reaction.

        Args:
            message_id: The id of the message to react to.
            emoji: A single emoji glyph (e.g. ``"👍"``, ``"❤"``, ``"🔥"``).
                Telegram restricts which emojis bots can use as reactions
                to a curated allowlist; unsupported emojis are rejected
                by the API.  Pass ``None`` to clear.
        """
        assert self._application is not None
        chat_id_int = _coerce_chat_id(chat_id)
        reaction = [ReactionTypeEmoji(emoji=emoji)] if emoji is not None else None
        await self._application.bot.set_message_reaction(
            chat_id=chat_id_int,
            message_id=message_id,
            reaction=reaction,
        )
        return {"status": "ok"}


_PHOTO_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm"})
_VOICE_EXTS = frozenset({".ogg", ".oga"})
_AUDIO_EXTS = frozenset({".mp3", ".m4a", ".wav", ".flac"})


def _classify(host_path: Path) -> str:
    ext = host_path.suffix.lower()
    if ext in _PHOTO_EXTS:
        return "photo"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _VOICE_EXTS:
        return "voice"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "document"


def _coerce_chat_id(chat_id: str) -> int:
    try:
        return int(chat_id)
    except ValueError as e:
        raise ValueError(f"telegram chat_id must be an integer; got {chat_id!r}") from e


def _prepare_text(text: str, parse_mode: str) -> tuple[str, str | None]:
    """Map a model-facing parse_mode to (body, ptb_parse_mode)."""
    if parse_mode not in _PARSE_MODE_TO_PTB:
        raise ValueError(f"telegram parse_mode must be 'plain' or 'html'; got {parse_mode!r}")
    if parse_mode == "html":
        return markdown_to_telegram_html(text), "HTML"
    return text, None


async def _send_single_media(
    bot: Any,
    *,
    chat_id: int,
    host_path: Path,
    caption: str | None,
    parse_mode: str | None = None,
) -> Any:
    kind = _classify(host_path)
    sender = {
        "photo": bot.send_photo,
        "video": bot.send_video,
        "voice": bot.send_voice,
        "audio": bot.send_audio,
        "document": bot.send_document,
    }[kind]
    kwargs: dict[str, Any] = {"chat_id": chat_id, kind: host_path}
    if caption is not None:
        kwargs["caption"] = caption
        if parse_mode is not None:
            kwargs["parse_mode"] = parse_mode
    return await sender(**kwargs)


def _build_media_group(
    host_paths: list[Path], *, caption: str | None, parse_mode: str | None = None
) -> list[Any]:
    # Caption rides on the FIRST item only — Telegram's API ignores
    # captions on items 2..N of a media group.
    items: list[Any] = []
    for idx, path in enumerate(host_paths):
        kind = _classify(path)
        kwargs: dict[str, Any] = {"media": path}
        if idx == 0 and caption is not None:
            kwargs["caption"] = caption
            if parse_mode is not None:
                kwargs["parse_mode"] = parse_mode
        if kind == "photo":
            items.append(InputMediaPhoto(**kwargs))
        elif kind == "video":
            items.append(InputMediaVideo(**kwargs))
        elif kind == "audio":
            items.append(InputMediaAudio(**kwargs))
        else:
            # Voice can't go in a media group; fall back to document.
            items.append(InputMediaDocument(**kwargs))
    return items


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
    if msg.edited:
        metadata["edit_of_message_id"] = msg.message_id
    if msg.sticker_emoji is not None:
        metadata["sticker_emoji"] = msg.sticker_emoji
    return metadata
