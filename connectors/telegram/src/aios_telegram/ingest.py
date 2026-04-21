"""aios ingest client.

POSTs inbound Telegram messages to ``/v1/connections/{id}/messages``. Retries
transient failures (network, 5xx) with exponential backoff; 4xx and
retry-exhaustion both log and drop. Missing messages are unrecoverable
(Telegram does not redeliver long-polling updates we've already acked by
advancing ``offset``), but the connector advances ``offset`` only after the
handler returns — so a crash mid-handler causes redelivery on the next run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Self

import httpx
import structlog

from .parse import InboundMessage

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


def build_metadata(msg: InboundMessage, bot_id: int) -> dict[str, Any]:
    # `channel` is redundant with what aios stamps server-side, but we
    # include it so events are self-describing when read outside aios.
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
