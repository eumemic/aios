"""``python -m aios_telegram`` entry point.

Reads ``AIOS_URL`` and ``AIOS_CONNECTOR_TOKEN`` from env (the SDK does
this automatically inside ``HttpConnector.__init__``).  The bot token
lives on the connection record (encrypted at rest server-side); the
connector fetches it via the SDK's ``self.secrets()`` helper at
``setup()`` time.
"""

from __future__ import annotations

import asyncio

from .connector import TelegramConnector


def main() -> None:
    asyncio.run(TelegramConnector().run())


if __name__ == "__main__":
    main()
