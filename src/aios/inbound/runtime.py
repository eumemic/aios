"""Long-running MCP inbound supervisor."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import asyncpg

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.logging import get_logger
from aios.models.inbound import InboundSubscriptionSpec
from aios.services import inbound as inbound_service
from aios.services.vaults import is_expiring, refresh_credential

from .protocol import (
    NOTIFICATION_CHANNELS_DELTA,
    NOTIFICATION_CHANNELS_SNAPSHOT,
    NOTIFICATION_MESSAGE,
    NOTIFICATION_REPLAY_LOST,
    RawInboundMcpClient,
    find_hidden_subscribe_tool,
    has_aios_inbound_capability,
)

log = get_logger("aios.inbound")

POLL_INTERVAL_SECONDS = 5.0
UNSUPPORTED_RETRY_SECONDS = 60.0
RECONNECT_BASE_SECONDS = 1.0
RECONNECT_MAX_SECONDS = 30.0


class InboundSupervisor:
    def __init__(
        self,
        *,
        pool: asyncpg.Pool[Any],
        crypto_box: CryptoBox,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self.pool = pool
        self.crypto_box = crypto_box
        self.poll_interval_seconds = poll_interval_seconds
        self._tasks: dict[tuple[str, str, str, str, str, str], asyncio.Task[None]] = {}

    async def run_forever(self) -> None:
        try:
            while True:
                await self.reconcile_once()
                await asyncio.sleep(self.poll_interval_seconds)
        finally:
            await self.shutdown()

    async def reconcile_once(self) -> None:
        async with self.pool.acquire() as conn:
            specs = await queries.list_inbound_subscription_specs(conn)
        desired = {spec.key: spec for spec in specs}

        for key, task in list(self._tasks.items()):
            if task.done() or key not in desired:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                self._tasks.pop(key, None)

        for key, spec in desired.items():
            if key in self._tasks:
                continue
            self._tasks[key] = asyncio.create_task(
                self._run_subscription(spec),
                name=f"mcp-inbound:{spec.session_id}:{spec.mcp_server_name}:{spec.account_id}",
            )

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_subscription(self, spec: InboundSubscriptionSpec) -> None:
        delay = RECONNECT_BASE_SECONDS
        while True:
            try:
                await self._subscription_once(spec)
                delay = RECONNECT_BASE_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning(
                    "inbound.subscription_failed",
                    session_id=spec.session_id,
                    server_name=spec.mcp_server_name,
                    account_id=spec.account_id,
                    exc_info=True,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_SECONDS)

    async def _subscription_once(self, spec: InboundSubscriptionSpec) -> None:
        headers = await _headers_for_spec(self.pool, self.crypto_box, spec)
        async with self.pool.acquire() as conn:
            since_event_id = await queries.get_inbound_cursor(
                conn,
                session_id=spec.session_id,
                mcp_server_name=spec.mcp_server_name,
                vault_credential_id=spec.vault_credential_id,
                account_id=spec.account_id,
            )

        async with RawInboundMcpClient(spec.mcp_server_url, headers) as client:
            init_result = await client.initialize()
            if not has_aios_inbound_capability(init_result):
                log.info(
                    "inbound.unsupported_server",
                    session_id=spec.session_id,
                    server_name=spec.mcp_server_name,
                    url=spec.mcp_server_url,
                )
                await asyncio.sleep(UNSUPPORTED_RETRY_SECONDS)
                return

            tools = await client.list_tools()
            if find_hidden_subscribe_tool(tools) is None:
                log.info(
                    "inbound.subscribe_tool_missing",
                    session_id=spec.session_id,
                    server_name=spec.mcp_server_name,
                    url=spec.mcp_server_url,
                )
                await asyncio.sleep(UNSUPPORTED_RETRY_SECONDS)
                return

            await client.call_tool(
                "aios_inbound_subscribe",
                {
                    "account_id": spec.account_id,
                    "since_event_id": since_event_id,
                },
            )
            log.info(
                "inbound.subscribed",
                session_id=spec.session_id,
                server_name=spec.mcp_server_name,
                account_id=spec.account_id,
                since_event_id=since_event_id,
            )

            async for notification in client.notifications():
                params = notification.params if isinstance(notification.params, dict) else {}
                await self._handle_notification(spec, notification.method, params)

    async def _handle_notification(
        self,
        spec: InboundSubscriptionSpec,
        method: str,
        params: dict[str, Any],
    ) -> None:
        if method == NOTIFICATION_MESSAGE:
            event_id = params.get("event_id")
            channel = params.get("channel")
            content = params.get("content")
            metadata = params.get("metadata") or {}
            if (
                not isinstance(event_id, str)
                or not isinstance(channel, str)
                or not isinstance(content, str)
            ):
                log.warning("inbound.bad_message_notification", method=method, params=params)
                return
            if not isinstance(metadata, dict):
                metadata = {}
            await inbound_service.ingest_inbound_message(
                self.pool,
                spec,
                event_id=event_id,
                channel=channel,
                content=content,
                metadata=metadata,
            )
            return

        if method == NOTIFICATION_CHANNELS_SNAPSHOT:
            channels = params.get("channels")
            await inbound_service.apply_channels_snapshot(
                self.pool,
                spec,
                channels if isinstance(channels, list) else [],
            )
            return

        if method == NOTIFICATION_CHANNELS_DELTA:
            upserts = params.get("upserts") or params.get("upsert") or []
            archives = (
                params.get("archives") or params.get("archived") or params.get("removes") or []
            )
            await inbound_service.apply_channels_delta(
                self.pool,
                spec,
                upserts=upserts if isinstance(upserts, list) else [],
                archives=archives if isinstance(archives, list) else [],
            )
            return

        if method == NOTIFICATION_REPLAY_LOST:
            await inbound_service.record_replay_lost(self.pool, spec, params)
            return

        log.debug("inbound.unknown_notification", method=method)


async def _headers_for_spec(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    spec: InboundSubscriptionSpec,
) -> dict[str, str]:
    blob = spec.blob
    auth_type = spec.auth_type
    payload = json.loads(crypto_box.decrypt(blob))
    if auth_type == "mcp_oauth" and is_expiring(payload):
        async with pool.acquire() as conn:
            await refresh_credential(
                crypto_box,
                conn,
                vault_id=spec.vault_id,
                mcp_server_url=spec.mcp_server_url,
            )
            refreshed = await queries.resolve_vault_credential(
                conn,
                vault_id=spec.vault_id,
                mcp_server_url=spec.mcp_server_url,
            )
            if refreshed is None:
                return {}
            blob, auth_type = refreshed
            payload = json.loads(crypto_box.decrypt(blob))

    token = str(payload.get("access_token" if auth_type == "mcp_oauth" else "token", ""))
    return {"Authorization": f"Bearer {token}"} if token else {}


async def inbound_main() -> None:
    from aios.config import get_settings
    from aios.db.pool import create_pool
    from aios.harness.procrastinate_app import app as procrastinate_app
    from aios.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    pool = await create_pool(settings.db_url, max_size=settings.db_pool_max_size)
    crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    await procrastinate_app.open_async()
    supervisor = InboundSupervisor(pool=pool, crypto_box=crypto_box)
    log.info("inbound.startup")
    try:
        await supervisor.run_forever()
    finally:
        log.info("inbound.shutdown")
        await supervisor.shutdown()
        await procrastinate_app.close_async()
        await pool.close()
