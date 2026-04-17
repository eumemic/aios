"""Top-level orchestration.

``run(cfg)`` supervises three tasks under one ``asyncio.TaskGroup``: the
signal-cli subprocess, the inbound pump (listener → parse → POST aios), and
the MCP server. Any task failure propagates through the group and exits the
process non-zero — operator systemd/Docker restarts.
"""

from __future__ import annotations

import asyncio
import contextlib

import structlog

from .config import Settings
from .daemon import SignalDaemon
from .ingest import InboundPump, IngestClient
from .mcp import build_mcp_app, build_mcp_server, parse_bind, serve_mcp

log = structlog.get_logger(__name__)


async def run(cfg: Settings) -> None:
    async with SignalDaemon(
        phone=cfg.phone,
        config_dir=cfg.config_dir,
        cli_bin=cfg.cli_bin,
        host=cfg.daemon_host,
        port=cfg.daemon_port,
    ) as daemon:
        bot_uuid = await daemon.discover_bot_uuid()
        log.info("signal.ready", bot_uuid=bot_uuid, phone=cfg.phone)

        async with IngestClient(
            base_url=cfg.aios_url,
            api_key=cfg.aios_api_key,
            connection_id=cfg.aios_connection_id,
        ) as ingest:
            pump = InboundPump(
                bot_uuid=bot_uuid,
                ingest=ingest,
                messages=daemon.listener.messages(),
            )
            mcp_app = build_mcp_app(build_mcp_server(rpc=daemon.rpc), token=cfg.mcp_token)
            host, port = parse_bind(cfg.mcp_bind)

            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(pump.run(), name="signal-pump")
                    tg.create_task(serve_mcp(mcp_app, host=host, port=port), name="signal-mcp")
                    tg.create_task(_await_crash(daemon), name="signal-crash-watch")
            except* Exception as eg:
                # Surface the first real exception so operators see a conventional
                # traceback rather than ExceptionGroup noise.
                raise eg.exceptions[0] from None


async def _await_crash(daemon: SignalDaemon) -> None:
    with contextlib.suppress(asyncio.CancelledError):
        await daemon.crashed()
