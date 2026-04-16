"""Top-level orchestration.

``run(cfg)`` wires the three moving parts of the connector into one
``asyncio.TaskGroup``:

1. :class:`SignalDaemon` owns the signal-cli subprocess (enter: spawn, wait
   for TCP, discover the bot's ACI UUID).
2. :class:`InboundPump` drains the listener, parses envelopes, and POSTs them
   to aios.
3. The FastMCP server (``signal_send``, ``signal_react``,
   ``signal_read_receipt``) is served on uvicorn.

Crash-is-fatal: any task failure propagates through the TaskGroup, tearing
the process down with a non-zero exit. Operator systemd/Docker restarts.
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
    """Blocking run loop — returns only on graceful shutdown or fatal crash."""
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
            mcp = build_mcp_server(rpc=daemon.rpc, bot_account_uuid=bot_uuid)
            mcp_app = build_mcp_app(mcp, token=cfg.mcp_token)
            host, port = parse_bind(cfg.mcp_bind)

            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(pump.run(), name="signal-pump")
                    tg.create_task(serve_mcp(mcp_app, host=host, port=port), name="signal-mcp")
                    tg.create_task(_await_crash(daemon), name="signal-crash-watch")
            except* Exception as eg:
                # Surface the first real exception so the caller sees a
                # conventional traceback rather than ExceptionGroup repr.
                raise eg.exceptions[0] from None


async def _await_crash(daemon: SignalDaemon) -> None:
    """Feed the daemon's crash future into the TaskGroup as a regular task."""
    with contextlib.suppress(asyncio.CancelledError):
        await daemon.crashed()
