"""Signal connector ported to the aios-connector SDK.

Replaces the pre-PR3 FastMCP HTTP server + ingest-HTTP-POST architecture
with a single :class:`aios_connector.Connector` subclass communicating
with aios over stdio MCP.

Multi-account: one signal-cli daemon serves N registered phones (multi-
account mode, no ``-a`` flag).  The connector aggregates per-phone
identity (uuid + contacts + groups) and routes every send / react RPC
through the explicit ``account`` param.  Single-phone setups still
work — set ``AIOS_SIGNAL_PHONES`` to a one-element list.

Lifecycle:

* :meth:`setup` opens :class:`SignalDaemon` (which spawns ``signal-cli
  daemon`` in multi-account mode and waits for TCP readiness),
  discovers the per-phone bot UUIDs, and loads contacts + groups for
  every account.
* :meth:`discover_accounts` returns one entry per configured phone.
* :meth:`serve` drives the inbound pump: drains ``(account, envelope)``
  pairs from ``daemon.listener``, parses them with the right per-phone
  bot UUID, and calls :meth:`emit_inbound` with the matching account.
  Spool durability + dedup ledger are handled by the SDK.
* :meth:`teardown` closes the daemon (SIGTERM → grace → SIGKILL).
* The two model-facing tools, ``signal_send`` and ``signal_react``,
  use :func:`focal_required` to receive ``account`` and ``chat_id``
  from ``_meta.aios.focal_channel_path`` and route the RPC accordingly.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

from .addressing import decode_chat_id, encode_chat_id
from .config import Settings
from .daemon import GroupInfo, SignalDaemon
from .markdown import convert_markdown_to_signal_styles
from .mentions import build_mention_strings, encode_mentions
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
        # Phone → bot UUID, populated during setup() from accounts.json.
        self._bot_uuids: dict[str, str] = {}
        # Convenience reverse map: UUID → phone, used in the inbound pump
        # to look up the bot identity for a per-account envelope parse.
        self._uuid_to_phone: dict[str, str] = {}
        # Contacts + group rosters are per-account because each phone has
        # its own contact store and group memberships in signal-cli.
        self._contact_names_by_account: dict[str, dict[str, str]] = {}
        # Group rosters retained for runtime use (outbound mention
        # encoding needs the focal group's member UUIDs at send time).
        self._groups_by_account: dict[str, list[GroupInfo]] = {}
        # Phone → error message for accounts whose boot-time probe
        # (``listContacts`` / ``listGroups``) failed.  signal-cli's
        # daemon stays running and accepts new attachments to other
        # accounts, so we don't refuse the connector outright; but
        # unhealthy accounts are dropped from ``discover_accounts``
        # so the operator's ``aios connectors accounts`` listing
        # surfaces only the working ones.  The connector refuses
        # to start if NO accounts are healthy (in :meth:`setup`).
        self._unavailable_accounts: dict[str, str] = {}

    # ── lifecycle ─────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Open the signal-cli daemon and load contacts + groups for every phone.

        signal-cli takes 5+ seconds to come up; the supervisor's bounded
        init handshake (30s) accommodates this.  The SDK doesn't
        finish ``initialize()`` until this method returns, so the
        supervisor stays in ``starting`` and aios won't dispatch tool
        calls against an unready daemon.
        """
        self._daemon = await SignalDaemon(
            phones=self._cfg.phones,
            config_dir=self._cfg.config_dir,
            cli_bin=self._cfg.cli_bin,
            host=self._cfg.daemon_host,
            port=self._cfg.daemon_port,
        ).__aenter__()
        self._bot_uuids = await self._daemon.discover_bot_uuids()
        self._uuid_to_phone = {uuid: phone for phone, uuid in self._bot_uuids.items()}
        # Load per-account contacts + groups in series — N is small
        # (typically 1-3 phones) and parallel listContacts calls would
        # race for the daemon's contact-store lock without measurable
        # speedup at this scale.
        instructions_sections: list[str] = []
        healthy_count = 0
        for phone, bot_uuid in self._bot_uuids.items():
            try:
                contact_names = await self._daemon.list_contacts(account=phone)
                groups = await self._daemon.list_groups(account=phone)
            except Exception as err:
                # signal-cli rejected the boot probe — typically because
                # the account isn't registered with Signal's servers
                # (operator hasn't run ``signal-cli register``, or the
                # registration expired).  Don't mark this account ready;
                # log loudly so the operator sees red on
                # ``aios connectors list`` instead of silent inbound-drops.
                log.error(
                    "signal.account.unavailable",
                    bot_uuid=bot_uuid,
                    phone=phone,
                    error=str(err),
                )
                self._unavailable_accounts[phone] = str(err)
                continue
            self._contact_names_by_account[phone] = contact_names
            self._groups_by_account[phone] = groups
            section = build_instructions(
                bot_uuid=bot_uuid,
                phone=phone,
                profile_name=contact_names.get(bot_uuid),
                groups=groups,
                contact_names=contact_names,
            )
            instructions_sections.append(section)
            healthy_count += 1
            log.info(
                "signal.account.ready",
                bot_uuid=bot_uuid,
                phone=phone,
                contacts=len(contact_names),
                groups=len(groups),
            )
        if healthy_count == 0 and self._bot_uuids:
            errs = ", ".join(f"{phone}: {err}" for phone, err in self._unavailable_accounts.items())
            raise RuntimeError(
                f"signal connector cannot start: no configured account is healthy ({errs})"
            )
        # Concatenate per-account sections.  build_instructions already
        # produces a self-contained block per account; joining with a
        # blank line gives the agent clearly delimited identities.
        self.instructions = "\n\n".join(instructions_sections) if instructions_sections else None

    async def discover_accounts(self) -> list[dict[str, Any]]:
        assert self._bot_uuids, "setup() must run before discover_accounts()"
        return [
            make_account(
                id=bot_uuid,
                display_name=self._contact_names_by_account.get(phone, {}).get(bot_uuid, phone),
                metadata={"phone": phone},
            )
            for phone, bot_uuid in self._bot_uuids.items()
            if phone not in self._unavailable_accounts
        ]

    async def teardown(self) -> None:
        if self._daemon is not None:
            await self._daemon.__aexit__(None, None, None)
            self._daemon = None

    async def serve(self) -> None:
        """Drain ``(account, envelope)`` pairs from signal-cli and emit to aios.

        Per-account routing: the listener stamps every receive
        notification with the phone the message arrived on; we look up
        the matching bot UUID and pass it to ``parse_envelope`` so
        self-message detection works correctly across all configured
        accounts.  Falls back on each account's contact store when an
        envelope's ``sourceName`` is empty.
        """
        assert self._daemon is not None, "setup() must run before serve()"
        async for account, envelope in self._daemon.listener.messages():
            phone = account.strip()
            bot_uuid = self._bot_uuids.get(phone)
            if bot_uuid is None:
                # Notification for an account we didn't register — most
                # likely operator added a phone via signal-cli directly
                # without restarting the connector.  Drop with a warning;
                # otherwise self-message filtering would misbehave.
                log.warning("signal.inbound.unknown_account", account=phone)
                continue
            msg = parse_envelope(envelope, bot_account_uuid=bot_uuid)
            if msg is None:
                continue
            contact_names = self._contact_names_by_account.get(phone, {})
            if msg.sender_name is None:
                resolved = contact_names.get(msg.sender_uuid)
                if resolved:
                    msg = replace(msg, sender_name=resolved)
            chat_id = encode_chat_id(msg.raw_chat_id, msg.chat_type)
            content = build_content_text(msg)
            metadata = build_metadata(msg, chat_id, bot_uuid)
            sender_payload: dict[str, Any] = {
                "uuid": msg.sender_uuid,
                "display_name": msg.sender_name or msg.sender_uuid,
            }
            sdk_attachments = self._build_sdk_attachments(msg)
            # Signal envelope timestamps are millis since epoch.  Render
            # them as ISO-8601 UTC so the supervisor stamps a string that
            # operators (and any future cross-platform tooling) can read
            # without knowing the connector's source-timestamp shape.
            timestamp_iso = (
                datetime.fromtimestamp(msg.timestamp_ms / 1000, tz=UTC).isoformat()
                if msg.timestamp_ms
                else None
            )
            await self.emit_inbound(
                account=bot_uuid,
                chat_id=chat_id,
                sender=sender_payload,
                content=content,
                attachments=sdk_attachments or None,
                metadata=metadata,
                timestamp=timestamp_iso,
            )

    # ── model-facing tools ────────────────────────────────────────────

    @tool()
    @focal_required
    async def signal_send(
        self,
        text: str,
        attachments: list[SandboxPath] | None = None,
        quote_timestamp_ms: int | None = None,
        quote_author_uuid: str | None = None,
        edit_timestamp_ms: int | None = None,
        *,
        account: str,
        chat_id: str,
    ) -> dict[str, Any]:
        """Send a Signal message to your focal chat, optionally with attachments.

        The account (your bot UUID) and chat id are taken implicitly
        from your focal channel — aios injects them via the JSON-RPC
        ``_meta`` field on each call.  Set focal with the built-in
        ``switch_channel`` tool.

        Args:
            text: Message body. Markdown is converted to Signal text styles.
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
        phone = self._resolve_phone(account, "signal_send")
        host_paths: list[Path] = list(attachments or [])
        member_uuids = await self._group_member_uuids(phone, chat_id)
        params = _build_send_params(
            phone,
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
    @focal_required
    async def signal_delete(
        self,
        target_timestamp_ms: int,
        *,
        account: str,
        chat_id: str,
    ) -> dict[str, Any]:
        """Delete-for-everyone a prior message in your focal Signal chat.

        signal-cli only allows deleting messages this account sent;
        attempting to delete someone else's message returns an RPC
        error.  Pass the ``sent_at_ms`` you got back from
        ``signal_send`` (or ``timestamp_ms`` from an inbound header
        for self-messages received as sync events).

        Args:
            target_timestamp_ms: The Signal timestamp of the message to delete.
        """
        assert self._daemon is not None
        phone = self._resolve_phone(account, "signal_delete")
        chat_type, raw_id = decode_chat_id(chat_id)
        params: dict[str, Any] = {
            "account": phone,
            "targetTimestamp": target_timestamp_ms,
        }
        if chat_type == "group":
            params["groupId"] = raw_id
        else:
            params["recipient"] = [raw_id]
        await self._daemon.rpc.call("remoteDelete", params)
        return {"status": "ok"}

    @tool()
    @focal_required
    async def signal_create_group(
        self,
        name: str,
        member_uuids: list[str],
        *,
        account: str,
        chat_id: str,
    ) -> dict[str, Any]:
        """Create a new Signal group on the focal account.

        The new group is owned by the focal account.  ``chat_id`` is
        passed through by the focal protocol but ignored here — the
        new group has no chat_id yet.

        Args:
            name: Group display name.
            member_uuids: ACI UUIDs of the members to add (excluding this
                account, which is added implicitly as the creator).
        """
        assert self._daemon is not None
        phone = self._resolve_phone(account, "signal_create_group")
        result = await self._daemon.rpc.call(
            "updateGroup",
            {"account": phone, "name": name, "members": member_uuids},
        )
        if not isinstance(result, dict) or not result.get("groupId"):
            raise RuntimeError(f"signal-cli updateGroup did not return a groupId; got {result!r}")
        # Refresh the roster cache so an immediate signal_send into this
        # new group can resolve ``@<uuid_prefix>`` mentions against its
        # members.  Without the refresh the next send would silently drop
        # any mentions (group missing from cache → empty member_uuids).
        self._groups_by_account[phone] = await self._daemon.list_groups(account=phone)
        return {"group_id": result["groupId"]}

    @tool()
    @focal_required
    async def signal_rename_group(
        self,
        name: str,
        *,
        account: str,
        chat_id: str,
    ) -> dict[str, Any]:
        """Rename your focal Signal group.

        Only valid when the focal channel is a group; calling this from
        a DM raises ``ValueError`` with a message the model can recover
        from on the next turn.

        Args:
            name: The new group name.
        """
        assert self._daemon is not None
        phone = self._resolve_phone(account, "signal_rename_group")
        chat_type, raw_id = decode_chat_id(chat_id)
        if chat_type != "group":
            raise ValueError("signal_rename_group: focal channel is a DM, not a group")
        await self._daemon.rpc.call(
            "updateGroup",
            {"account": phone, "groupId": raw_id, "name": name},
        )
        return {"status": "ok"}

    def _resolve_phone(self, account: str, tool_name: str) -> str:
        phone = self._uuid_to_phone.get(account)
        if phone is None:
            raise ValueError(f"{tool_name}: unknown account {account!r}")
        return phone

    async def _group_member_uuids(self, phone: str, chat_id: str) -> list[str]:
        """Member UUIDs of the focal group, or ``[]`` for a DM.

        On cache miss, refreshes the per-account roster once via
        ``listGroups`` and re-checks.  signal-cli sometimes returns an
        empty group list at boot before the account's group state has
        finished loading from disk; once the cache is empty the bot
        never re-checks, so mentions stay broken for the lifetime of
        the connector.  Refreshing on miss also picks up groups joined
        after boot.
        """
        chat_type, _ = decode_chat_id(chat_id)
        if chat_type != "group":
            return []
        cached = self._lookup_group_members(phone, chat_id)
        if cached is not None:
            return cached
        assert self._daemon is not None
        self._groups_by_account[phone] = await self._daemon.list_groups(account=phone)
        return self._lookup_group_members(phone, chat_id) or []

    def _lookup_group_members(self, phone: str, chat_id: str) -> list[str] | None:
        for group in self._groups_by_account.get(phone, []):
            if group.id == chat_id:
                return list(group.member_uuids)
        return None

    @tool()
    @focal_required
    async def signal_react(
        self,
        target_author_uuid: str,
        target_timestamp_ms: int,
        emoji: str,
        *,
        account: str,
        chat_id: str,
    ) -> dict[str, Any]:
        """React to a message in your focal Signal chat with an emoji.

        The account (your bot UUID) and chat id are taken implicitly
        from your focal channel — aios injects them via the JSON-RPC
        ``_meta`` field on each call.

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
        phone = self._resolve_phone(account, "signal_react")
        params = _build_react_params(phone, chat_id, target_author_uuid, target_timestamp_ms, emoji)
        await self._daemon.rpc.call("sendReaction", params)
        return {"status": "ok"}

    def _build_sdk_attachments(self, msg: InboundMessage) -> list[SDKAttachment]:
        """Build SDK Attachment records, logging+skipping any rejected.

        signal-cli's JSON-RPC daemon mode (0.14.x) auto-downloads
        attachment bytes but omits the ``file`` field from the
        envelope — it was only emitted by the legacy CLI command
        output.  We fall back to the storage-layout convention
        ``<config_dir>/attachments/<id>`` for envelopes without a
        host_path; SDK ``as_params`` validates the file actually
        exists.
        """
        out: list[SDKAttachment] = []
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
                candidate.as_params()
            except AttachmentError as err:
                log.warning(
                    "signal.inbound.attachment_rejected",
                    host_path=host_path,
                    filename=att.filename,
                    error=str(err),
                )
                continue
            out.append(candidate)
        return out


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
    account at that timestamp.  signal-cli routes via the same ``send``
    RPC; the timestamp identifies which message to replace.
    """
    if (quote_timestamp_ms is None) != (quote_author_uuid is None):
        raise ValueError("quote_timestamp_ms and quote_author_uuid must be set together")
    chat_type, raw_id = decode_chat_id(chat_id)
    # Mention encoding inserts U+FFFC placeholders; markdown stripping
    # then removes delimiter chars (which can shift placeholder offsets
    # leftward).  Compute Signal's mention offsets against the *final*
    # stripped message so they match what the recipient sees.
    encoded, ordered_uuids = encode_mentions(text, member_uuids or [])
    stripped, styles = convert_markdown_to_signal_styles(encoded)
    mentions = build_mention_strings(stripped, ordered_uuids)
    params: dict[str, Any] = {"account": account_phone, "message": stripped}
    if styles:
        params["textStyles"] = styles
    if mentions:
        params["mentions"] = mentions
    if chat_type == "group":
        params["groupId"] = raw_id
    else:
        params["recipient"] = [raw_id]
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
    chat_type, raw_id = decode_chat_id(chat_id)
    params: dict[str, Any] = {
        "account": account_phone,
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
