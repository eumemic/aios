"""``python -m aios_telegram`` entry point.

Reads ``AIOS_URL`` and ``AIOS_RUNTIME_TOKEN`` from env (the SDK does
this automatically inside ``HttpConnector.__init__``).  Each bot token
lives on its connection record (encrypted at rest server-side); the
connector fetches secrets per-connection inside ``serve_connection``.
"""

from __future__ import annotations

import asyncio

from .connector import TelegramConnector


def main() -> None:
    asyncio.run(TelegramConnector().run_until_stopped())


if __name__ == "__main__":
    main()
