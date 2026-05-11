"""Signal connector built on the aios-connector-http SDK.

Single-phone-per-container: each connector container runs its own
signal-cli daemon serving one registered phone.  Multi-phone
deployments use multiple containers, each with its own connector
token.  The phone (account identity) lives on the connection
record's encrypted secrets and is fetched at ``setup()`` time via
``self.secrets()``.

Lifecycle:

* :meth:`setup` opens :class:`SignalDaemon` (a one-element phones list
  is the unchanged contract with the daemon module), discovers the
  bot UUID, and loads contacts + groups.
* :meth:`serve` drains envelopes from ``daemon.listener``, parses
  each, and forwards to :meth:`emit_inbound`.
* :meth:`teardown` closes the daemon.

The two model-facing tools take ``chat_id`` from the call's
``focal_channel`` automatically — declare it as a kwarg and the SDK
threads it through.  ``account`` (the bot's signal UUID) is implicit:
each container is one account, so tool methods read ``self._bot_uuid``
directly rather than receiving it.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from aios_connector_http import (
    Attachment as SDKAttachment,
)
from aios_connector_http import (
    AttachmentError,
    HttpConnector,
    SandboxPath,
    tool,
)

from .addressing import decode_chat_id, encode_chat_id
from .config import Settings
from .daemon import GroupInfo, SignalDaemon
from .markdown import convert_markdown_to_signal_styles
from .mentions import build_mention_strings, encode_mentions
from .parse import InboundMessage, build_content_text, is_group_update_envelope, parse_envelope

log = structlog.get_logger(__name__)


class SignalConnector(HttpConnector):
    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self._cfg = cfg
        self._daemon: SignalDaemon | None = None
        self._bot_uuid: str | None = None
        self._phone: str | None = None
        self._contact_names: dict[str, str] = {}
        self._groups: list[GroupInfo] = []

    # ── lifecycle ─────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Open the signal-cli daemon and load the bot's contacts + groups.

        signal-cli takes 5+ seconds to come up; the daemon module's
        bounded TCP-readiness loop accommodates this.
        """
        secrets = await self.secrets()
        phone = secrets.get("phone")
        if not phone:
            raise RuntimeError(
                "signal connector requires a 'phone' entry in its connection's secrets"
            )
        self._phone = phone
        self._daemon = await SignalDaemon(
            phones=[phone],
            config_dir=self._cfg.config_dir,
            cli_bin=self._cfg.cli_bin,
            host=self._cfg.daemon_host,
            port=self._cfg.daemon_port,
        ).__aenter__()
        bot_uuids = await self._daemon.discover_bot_uuids()
        self._bot_uuid = bot_uuids[phone]
        self._contact_names = await self._daemon.list_contacts(account=phone)
        self._groups = await self._daemon.list_groups(account=phone)
        log.info(
            "signal.account.ready",
            bot_uuid=self._bot_uuid,
            phone=phone,
            contacts=len(self._contact_names),
            groups=len(self._groups),
        )

    async def teardown(self) -> None:
        if self._daemon is not None:
            await self._daemon.__aexit__(None, None, None)
            self._daemon = None

    async def serve(self) -> None:
        """Drain envelopes from signal-cli and emit each as an aios inbound.

        signal-cli stamps every receive notification with ``account``
        (the phone).  Since this daemon serves only our phone, every
        notification's account matches ``self._phone``; envelopes for
        other accounts shouldn't appear, but we filter anyway.
        """
        assert self._daemon is not None, "setup() must run before serve()"
        assert self._bot_uuid is not None
        async for account, envelope in self._daemon.listener.messages():
            if account.strip() != self._phone:
                log.warning("signal.inbound.unknown_account", account=account)
                continue
            await self._maybe_refresh_roster(envelope)
            msg = parse_envelope(envelope, bot_account_uuid=self._bot_uuid)
            if msg is None:
                continue
            if msg.sender_name is None:
                resolved = self._contact_names.get(msg.sender_uuid)
                if resolved:
                    msg = replace(msg, sender_name=resolved)
            chat_id = encode_chat_id(msg.raw_chat_id, msg.chat_type)
            sender_payload: dict[str, Any] = {
                "uuid": msg.sender_uuid,
                "display_name": msg.sender_name or msg.sender_uuid,
            }
            attachments = self._build_attachment_dicts(msg)
            # Signal envelope timestamps are ms since epoch.  Render as
            # ISO-8601 UTC so operators reading event logs see absolute
            # times rather than connector-source unix-ms.
            timestamp_iso = (
                datetime.fromtimestamp(msg.timestamp_ms / 1000, tz=UTC).isoformat()
                if msg.timestamp_ms
                else None
            )
            await self.emit_inbound(
                chat_id=chat_id,
                sender=sender_payload,
                content=build_content_text(msg),
                attachments=attachments or None,
                metadata=build_metadata(msg, chat_id, self._bot_uuid),
                timestamp=timestamp_iso,
            )

    # ── model-facing tools ────────────────────────────────────────────

    @tool()
    async def signal_send(
        self,
        text: str,
        attachments: list[SandboxPath] | None = None,
        quote_timestamp_ms: int | None = None,
        quote_author_uuid: str | None = None,
        edit_timestamp_ms: int | None = None,
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """Send a Signal message to your focal chat, optionally with attachments.

        The chat id is taken implicitly from your focal channel — the
        SDK injects it from the call's ``focal_channel``.

        Args:
            text: Message body.  Markdown is converted to Signal text styles.
                In group chats, ``@<uuid_prefix>`` mentions are encoded
                automatically when the prefix uniquely matches a member.
            attachments: Optional in-sandbox file paths to attach.  The SDK
                resolves each entry to a host path before this method runs.
            quote_timestamp_ms: ``timestamp_ms`` of the message you're
                quoting.  Must be set together with ``quote_author_uuid``;
                passing only one raises ``ValueError``.
            quote_author_uuid: ``sender_uuid`` of the message you're quoting.
            edit_timestamp_ms: ``sent_at_ms`` of a prior message FROM this
                account to rewrite.  The new ``text`` replaces the old one;
                Signal clients render an "edited" indicator.
        """
        assert self._daemon is not None
        assert self._phone is not None
        host_paths: list[Path] = list(attachments or [])
        member_uuids = await self._group_member_uuids(chat_id)
        params = _build_send_params(
            self._phone,
            chat_id,
            text,
            attachments=host_paths,
            member_uuids=member_uuids,
            quote_timestamp_ms=quote_timestamp_ms,
            quote_author_uuid=quote_author_uuid,
            edit_timestamp_ms=edit_timestamp_ms,
        )
        result = await self._daemon.rpc.call("send", params)
        ts = _extract_timestamp(result)
        return {"sent_at_ms": ts} if ts is not None else {"status": "ok"}

    @tool()
    async def signal_delete(
        self,
        target_timestamp_ms: int,
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """Delete-for-everyone a prior message in your focal Signal chat.

        signal-cli only allows deleting messages this account sent.

        Args:
            target_timestamp_ms: The Signal timestamp of the message to delete.
        """
        assert self._daemon is not None
        params: dict[str, Any] = {
            "account": self._phone,
            "targetTimestamp": target_timestamp_ms,
            **_recipient_params(chat_id),
        }
        await self._daemon.rpc.call("remoteDelete", params)
        return {"status": "ok"}

    @tool()
    async def signal_create_group(
        self,
        name: str,
        member_uuids: list[str],
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """Create a new Signal group on this account.

        ``chat_id`` is passed through by the focal protocol but ignored
        here — the new group has no chat_id yet.

        Args:
            name: Group display name.
            member_uuids: ACI UUIDs of the members to add (excluding this
                account, which is added implicitly as the creator).
        """
        assert self._daemon is not None
        result = await self._daemon.rpc.call(
            "updateGroup",
            {"account": self._phone, "name": name, "members": member_uuids},
        )
        if not isinstance(result, dict) or not result.get("groupId"):
            raise RuntimeError(f"signal-cli updateGroup did not return a groupId; got {result!r}")
        # Refresh roster so an immediate signal_send into the new group
        # can resolve ``@<uuid_prefix>`` mentions against its members.
        await self._refresh_roster()
        return {"group_id": result["groupId"]}

    @tool()
    async def signal_rename_group(
        self,
        name: str,
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """Rename your focal Signal group.

        Only valid when the focal channel is a group.

        Args:
            name: The new group name.
        """
        assert self._daemon is not None
        chat_type, raw_id = decode_chat_id(chat_id)
        if chat_type != "group":
            raise ValueError("signal_rename_group: focal channel is a DM, not a group")
        await self._daemon.rpc.call(
            "updateGroup",
            {"account": self._phone, "groupId": raw_id, "name": name},
        )
        return {"status": "ok"}

    @tool()
    async def signal_react(
        self,
        target_author_uuid: str,
        target_timestamp_ms: int,
        emoji: str,
        *,
        chat_id: str,
    ) -> dict[str, Any]:
        """React to a message in your focal Signal chat with an emoji.

        The target message is identified by ``(target_author_uuid,
        target_timestamp_ms)``.  Every inbound Signal message in your
        conversation starts with a header line like ``[channel=... ·
        from=... · sender_uuid=<uuid> · timestamp_ms=<ms> (<iso>)]``;
        copy ``sender_uuid`` and the raw ``timestamp_ms`` integer from
        that header.

        Args:
            target_author_uuid: ``sender_uuid`` of the target message.
            target_timestamp_ms: ``timestamp_ms`` of the target message.
            emoji: The reaction emoji.
        """
        assert self._daemon is not None
        assert self._phone is not None
        params = _build_react_params(
            self._phone, chat_id, target_author_uuid, target_timestamp_ms, emoji
        )
        await self._daemon.rpc.call("sendReaction", params)
        return {"status": "ok"}

    # ── helpers ───────────────────────────────────────────────────────

    async def _maybe_refresh_roster(self, envelope: dict[str, Any]) -> None:
        if is_group_update_envelope(envelope):
            await self._refresh_roster()

    async def _refresh_roster(self) -> None:
        assert self._daemon is not None
        assert self._phone is not None
        self._groups = await self._daemon.list_groups(account=self._phone)

    async def _group_member_uuids(self, chat_id: str) -> list[str]:
        """Member UUIDs of the focal group, or ``[]`` for a DM.

        On cache miss, refreshes the roster once via ``listGroups`` and
        re-checks — signal-cli sometimes returns an empty group list at
        boot before the account's group state has finished loading from
        disk; without refresh-on-miss, mentions stay broken for the
        connector's lifetime.  Refreshing on miss also picks up groups
        joined after boot.
        """
        chat_type, _ = decode_chat_id(chat_id)
        if chat_type != "group":
            return []
        cached = self._lookup_group_members(chat_id)
        if cached is not None:
            return cached
        await self._refresh_roster()
        return self._lookup_group_members(chat_id) or []

    def _lookup_group_members(self, chat_id: str) -> list[str] | None:
        for group in self._groups:
            if group.id == chat_id:
                return list(group.member_uuids)
        return None

    def _build_attachment_dicts(self, msg: InboundMessage) -> list[dict[str, Any]]:
        """Build wire-shape attachment records, logging+skipping any rejected.

        signal-cli's JSON-RPC daemon mode (0.14.x) auto-downloads
        attachment bytes but omits the ``file`` field — we fall back to
        the storage-layout convention ``<config_dir>/attachments/<id>``
        for envelopes without a ``host_path``.  SDK ``as_params``
        validates the file exists and is under the 5 MiB cap.
        """
        out: list[dict[str, Any]] = []
        for att in msg.attachments:
            host_path = att.host_path
            if host_path is None and att.id is not None:
                host_path = str(self._cfg.config_dir / "attachments" / att.id)
            if host_path is None:
                log.warning(
                    "signal.inbound.attachment_no_host_path",
                    content_type=att.content_type,
                    filename=att.filename,
                )
                continue
            candidate = SDKAttachment(
                host_path=host_path,
                filename=att.filename or "unnamed",
                content_type=att.content_type,
            )
            try:
                out.append(candidate.as_params())
            except AttachmentError as err:
                log.warning(
                    "signal.inbound.attachment_rejected",
                    host_path=host_path,
                    filename=att.filename,
                    error=str(err),
                )
                continue
        return out


def _recipient_params(chat_id: str) -> dict[str, Any]:
    """signal-cli's per-chat addressing: ``groupId`` for groups, ``recipient`` array for DMs."""
    chat_type, raw_id = decode_chat_id(chat_id)
    if chat_type == "group":
        return {"groupId": raw_id}
    return {"recipient": [raw_id]}


def _build_send_params(
    account_phone: str,
    chat_id: str,
    text: str,
    *,
    attachments: list[Path],
    member_uuids: list[str] | None = None,
    quote_timestamp_ms: int | None = None,
    quote_author_uuid: str | None = None,
    edit_timestamp_ms: int | None = None,
) -> dict[str, Any]:
    """Translate ``(account_phone, chat_id, text)`` into signal-cli ``send`` params.

    ``member_uuids`` enables outbound mention encoding for group sends:
    any ``@<uuid_prefix>`` in ``text`` that uniquely matches a member's
    UUID is rewritten as a U+FFFC placeholder + ``mentions`` array
    entry.  Mention encoding runs BEFORE markdown conversion so the
    UTF-16 offsets stay correct for both annotation lists.

    Quote: ``quote_timestamp_ms`` and ``quote_author_uuid`` must be
    set together; one-without-the-other raises ``ValueError`` (signal-cli
    rejects partial quotes, and silently dropping the model's intent
    leaves no signal that the thread didn't land).

    Edit: ``edit_timestamp_ms`` rewrites the prior message of this
    account at that timestamp.
    """
    if (quote_timestamp_ms is None) != (quote_author_uuid is None):
        raise ValueError("quote_timestamp_ms and quote_author_uuid must be set together")
    # Mention encoding inserts U+FFFC placeholders; markdown stripping
    # then removes delimiter chars (which can shift placeholder offsets
    # leftward).  Compute Signal's mention offsets against the *final*
    # stripped message so they match what the recipient sees.
    encoded, ordered_uuids = encode_mentions(text, member_uuids or [])
    stripped, styles = convert_markdown_to_signal_styles(encoded)
    mentions = build_mention_strings(stripped, ordered_uuids)
    params: dict[str, Any] = {
        "account": account_phone,
        "message": stripped,
        **_recipient_params(chat_id),
    }
    if styles:
        params["textStyles"] = styles
    if mentions:
        params["mentions"] = mentions
    if attachments:
        params["attachments"] = [str(p) for p in attachments]
    if quote_timestamp_ms is not None and quote_author_uuid is not None:
        params["quoteTimestamp"] = quote_timestamp_ms
        params["quoteAuthor"] = quote_author_uuid
    if edit_timestamp_ms is not None:
        params["editTimestamp"] = edit_timestamp_ms
    return params


def _build_react_params(
    account_phone: str,
    chat_id: str,
    target_author_uuid: str,
    target_timestamp_ms: int,
    emoji: str,
) -> dict[str, Any]:
    """Translate a react request into signal-cli ``sendReaction`` params."""
    return {
        "account": account_phone,
        "emoji": emoji,
        "targetAuthor": target_author_uuid,
        "targetTimestamp": target_timestamp_ms,
        **_recipient_params(chat_id),
    }


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
    have an ID to hand back.
    """
    if not isinstance(rpc_result, dict):
        return None
    ts = rpc_result.get("timestamp")
    return ts if isinstance(ts, int) else None
