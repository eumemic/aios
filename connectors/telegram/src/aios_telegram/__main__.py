"""CLI entry point.

``python -m aios_telegram start`` launches the connector. CLI flags override
env vars; env vars override defaults.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from . import app
from .config import Settings
from .logging import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aios_telegram")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start the telegram connector")
    start.add_argument("--bot-token", help="Telegram bot token (AIOS_TELEGRAM_BOT_TOKEN)")
    start.add_argument("--aios-url", help="aios base URL (AIOS_URL)")
    start.add_argument("--aios-api-key", help="aios bearer token (AIOS_API_KEY)")
    start.add_argument("--aios-connection-id", help="aios connection id (AIOS_CONNECTION_ID)")
    start.add_argument("--mcp-bind", help="MCP host:port (AIOS_TELEGRAM_MCP_BIND)")
    start.add_argument("--mcp-token", help="MCP bearer token (AIOS_TELEGRAM_MCP_TOKEN)")
    return parser


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    mapping = {
        "bot_token": "AIOS_TELEGRAM_BOT_TOKEN",
        "aios_url": "AIOS_URL",
        "aios_api_key": "AIOS_API_KEY",
        "aios_connection_id": "AIOS_CONNECTION_ID",
        "mcp_bind": "AIOS_TELEGRAM_MCP_BIND",
        "mcp_token": "AIOS_TELEGRAM_MCP_TOKEN",
    }
    for attr, env in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env] = str(value)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "start":
        parser.error(f"unknown command: {args.command}")

    _apply_cli_overrides(args)
    cfg = Settings()  # fields are populated from env / CLI overrides

    try:
        asyncio.run(app.run(cfg))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
