"""Signal-side MCP inbound broker."""

from __future__ import annotations

import asyncio
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any

import structlog
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

from .addressing import is_dm_chat_id
from .daemon import GroupInfo

log = structlog.get_logger(__name__)

NOTIFICATION_MESSAGE = "notifications/aios/inbound/message"
NOTIFICATION_CHANNELS_SNAPSHOT = "notifications/aios/inbound/channels_snapshot"
NOTIFICATION_CHANNELS_DELTA = "notifications/aios/inbound/channels_delta"
NOTIFICATION_REPLAY_LOST = "notifications/aios/inbound/replay_lost"

REPLAY_LIMIT = 512


@dataclass(frozen=True, slots=True)
class InboundEvent:
    event_id: str
    channel: str
    content: str
    metadata: dict[str, Any]


def signal_event_id(*, bot_uuid: str, path: str, metadata: dict[str, Any]) -> str:
    sender = metadata.get("sender_uuid") or ""
    timestamp = metadata.get("timestamp_ms") or ""
    raw = f"{bot_uuid}|{path}|{sender}|{timestamp}"
    return "sig_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def initial_signal_channels(
    *,
    bot_uuid: str,
    groups: list[GroupInfo],
    contact_names: dict[str, str],
) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    for uuid, name in sorted(contact_names.items()):
        if uuid == bot_uuid or not is_dm_chat_id(uuid):
            continue
        channels.append(
            {
                "channel": f"{bot_uuid}/{uuid}",
                "display_name": name or uuid,
                "metadata": {"chat_type": "dm", "sender_uuid": uuid},
            }
        )
    for group in groups:
        channels.append(
            {
                "channel": f"{bot_uuid}/{group.id}",
                "display_name": group.name or group.id,
                "metadata": {
                    "chat_type": "group",
                    "chat_name": group.name,
                    "member_uuids": group.member_uuids,
                },
            }
        )
    return channels


class SignalInboundBroker:
    """Fan Signal inbound events out to subscribed MCP sessions."""

    def __init__(
        self,
        *,
        bot_uuid: str,
        initial_channels: list[dict[str, Any]] | None = None,
        replay_limit: int = REPLAY_LIMIT,
    ) -> None:
        self.bot_uuid = bot_uuid
        self.replay_limit = replay_limit
        self._subscribers: dict[int, Any] = {}
        self._replay: deque[InboundEvent] = deque(maxlen=replay_limit)
        self._channels: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        for entry in initial_channels or []:
            channel = entry.get("channel")
            if isinstance(channel, str):
                self._channels[channel] = dict(entry)

    async def subscribe(self, *, account_id: str, since_event_id: str | None, session: Any) -> dict[str, Any]:
        if account_id != self.bot_uuid:
            raise ValueError(f"unknown Signal account {account_id!r}")
        async with self._lock:
            self._subscribers[id(session)] = session
            snapshot = list(self._channels.values())
            replay = self._events_after_locked(since_event_id)
            replay_lost = since_event_id is not None and replay is None

        await self._send(
            session,
            NOTIFICATION_CHANNELS_SNAPSHOT,
            {"channels": snapshot},
        )
        if replay_lost:
            await self._send(
                session,
                NOTIFICATION_REPLAY_LOST,
                {"since_event_id": since_event_id},
            )
            replay = []
        for event in replay or []:
            await self._send_event(session, event)
        return {
            "status": "subscribed",
            "channels": len(snapshot),
            "replayed": len(replay or []),
            "replay_lost": replay_lost,
        }

    async def post_message(
        self,
        *,
        path: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        event = InboundEvent(
            event_id=signal_event_id(bot_uuid=self.bot_uuid, path=path, metadata=metadata),
            channel=f"{self.bot_uuid}/{path}",
            content=content,
            metadata=dict(metadata),
        )
        display_name = _display_name(metadata, path)
        channel_metadata: dict[str, Any] = {
            k: v
            for k, v in metadata.items()
            if k in {"chat_type", "chat_name", "sender_uuid", "sender_name"}
        }
        channel_entry: dict[str, Any] = {
            "channel": event.channel,
            "display_name": display_name,
            "metadata": channel_metadata,
        }

        async with self._lock:
            existing = self._channels.get(event.channel)
            is_new = existing is None
            existing_metadata = (existing or {}).get("metadata")
            if not isinstance(existing_metadata, dict):
                existing_metadata = {}
            self._channels[event.channel] = {
                **(existing or {}),
                **channel_entry,
                "metadata": {**existing_metadata, **channel_metadata},
            }
            self._replay.append(event)
            subscribers = list(self._subscribers.values())

        if is_new:
            await self._broadcast(
                NOTIFICATION_CHANNELS_DELTA,
                {"upserts": [self._channels[event.channel]], "archives": []},
                subscribers,
            )
        await self._broadcast_event(event, subscribers)

    def _events_after_locked(self, since_event_id: str | None) -> list[InboundEvent] | None:
        if since_event_id is None:
            return []
        events = list(self._replay)
        for idx, event in enumerate(events):
            if event.event_id == since_event_id:
                return events[idx + 1 :]
        return None if events else []

    async def _broadcast(
        self,
        method: str,
        params: dict[str, Any],
        subscribers: list[Any],
    ) -> None:
        dead: list[int] = []
        for session in subscribers:
            try:
                await self._send(session, method, params)
            except Exception:
                dead.append(id(session))
                log.warning("signal.inbound.send_failed", exc_info=True)
        if dead:
            async with self._lock:
                for key in dead:
                    self._subscribers.pop(key, None)

    async def _broadcast_event(self, event: InboundEvent, subscribers: list[Any]) -> None:
        dead: list[int] = []
        for session in subscribers:
            try:
                await self._send_event(session, event)
            except Exception:
                dead.append(id(session))
                log.warning("signal.inbound.send_failed", exc_info=True)
        if dead:
            async with self._lock:
                for key in dead:
                    self._subscribers.pop(key, None)

    async def _send_event(self, session: Any, event: InboundEvent) -> None:
        await self._send(
            session,
            NOTIFICATION_MESSAGE,
            {
                "event_id": event.event_id,
                "channel": event.channel,
                "content": event.content,
                "metadata": event.metadata,
            },
        )

    async def _send(self, session: Any, method: str, params: dict[str, Any]) -> None:
        notification = JSONRPCNotification(jsonrpc="2.0", method=method, params=params)
        await session.send_message(SessionMessage(message=JSONRPCMessage(notification)))


def _display_name(metadata: dict[str, Any], path: str) -> str:
    for key in ("chat_name", "sender_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return path
