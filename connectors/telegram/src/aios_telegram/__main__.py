"""``python -m aios_telegram`` entry point.

Reads ``AIOS_URL`` and ``AIOS_CONNECTOR_TOKEN`` from env (the SDK does
this automatically inside ``HttpConnector.__init__``).  Telegram-specific
config (``AIOS_TELEGRAM_BOT_TOKEN``) feeds pydantic-settings.
"""

from __future__ import annotations

import asyncio

from .config import Settings
from .connector import TelegramConnector


def main() -> None:
    asyncio.run(TelegramConnector(Settings()).run())


if __name__ == "__main__":
    main()
