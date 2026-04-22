"""Top-level orchestration.

``run(cfg)`` supervises two tasks under one ``asyncio.TaskGroup``: the PTB
``Application`` driving the Telegram long-polling loop and posting inbound
messages to aios, and the MCP server serving ``telegram_send``. Any task
failure propagates through the group and exits the process non-zero —
operator systemd/Docker restarts.
"""

from __future__ import annotations

import asyncio

import structlog

from .bot import build_application, discover_bot_identity, install_handler, run_application
from .config import Settings
from .ingest import IngestClient
from .mcp import build_mcp_app, build_mcp_server, parse_bind, serve_mcp

log = structlog.get_logger(__name__)


async def run(cfg: Settings) -> None:
    application = build_application(cfg.bot_token)
    await application.initialize()
    try:
        identity = await discover_bot_identity(application)
    except BaseException:
        await application.shutdown()
        raise

    async with IngestClient(
        base_url=cfg.aios_url,
        api_key=cfg.aios_api_key,
        connection_id=cfg.aios_connection_id,
    ) as ingest:
        install_handler(application, bot_id=identity.id, ingest=ingest)
        mcp_app = build_mcp_app(
            build_mcp_server(
                bot=application.bot,
                bot_id=identity.id,
                first_name=identity.first_name,
                username=identity.username,
            ),
            token=cfg.mcp_token,
        )
        host, port = parse_bind(cfg.mcp_bind)

        log.info("telegram.ready", bot_id=identity.id)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(run_application(application), name="telegram-bot")
                tg.create_task(serve_mcp(mcp_app, host=host, port=port), name="telegram-mcp")
        except* Exception as eg:
            # Surface the first real exception so operators see a conventional
            # traceback rather than ExceptionGroup noise.
            raise eg.exceptions[0] from None
