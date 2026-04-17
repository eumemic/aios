"""aios ingest client + InboundPump.

POSTs inbound Signal messages to ``/v1/connections/{id}/messages``. Retries
transient failures (network, 5xx) with exponential backoff; 4xx and
retry-exhaustion both log and drop, and we let Signal's own redelivery on
reconnect recover anything not acked by the daemon.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Self

import httpx
import structlog

from .addressing import encode_chat_id
from .parse import InboundMessage, build_content_text, parse_envelope

log = structlog.get_logger(__name__)

RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)


@dataclass(slots=True)
class IngestClient:
    base_url: str
    api_key: str
    connection_id: str
    _client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def post_message(
        self,
        *,
        path: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        if self._client is None:
            raise RuntimeError("IngestClient must be used as an async context manager")
        url = f"/v1/connections/{self.connection_id}/messages"
        body = {"path": path, "content": content, "metadata": metadata}

        for attempt, delay in enumerate((0.0, *RETRY_DELAYS_SECONDS)):
            if delay:
                await asyncio.sleep(delay)
            try:
                response = await self._client.post(url, json=body)
            except httpx.HTTPError as e:
                log.warning("ingest.network_error", attempt=attempt, error=str(e), path=path)
                continue

            if response.is_success:
                return
            if 400 <= response.status_code < 500:
                # 4xx is our bug — retrying won't help.
                log.error(
                    "ingest.client_error",
                    status=response.status_code,
                    body=response.text[:500],
                    path=path,
                )
                return
            log.warning(
                "ingest.server_error",
                attempt=attempt,
                status=response.status_code,
                body=response.text[:500],
                path=path,
            )

        log.error("ingest.retries_exhausted", path=path)


def build_metadata(
    msg: InboundMessage,
    chat_id: str,
    bot_uuid: str,
) -> dict[str, Any]:
    # `channel` is redundant with what aios stamps server-side, but we
    # include it so events are self-describing when read outside aios.
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


@dataclass(slots=True)
class InboundPump:
    bot_uuid: str
    ingest: IngestClient
    messages: AsyncIterator[dict[str, Any]]

    async def run(self) -> None:
        async for envelope in self.messages:
            msg = parse_envelope(envelope, bot_account_uuid=self.bot_uuid)
            if msg is None:
                continue
            chat_id = encode_chat_id(msg.raw_chat_id, msg.chat_type)
            content = build_content_text(msg)
            metadata = build_metadata(msg, chat_id, self.bot_uuid)
            await self.ingest.post_message(path=chat_id, content=content, metadata=metadata)
