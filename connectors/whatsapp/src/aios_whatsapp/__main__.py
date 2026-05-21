"""``python -m aios_whatsapp`` entry point."""

from __future__ import annotations

import asyncio

from .config import Settings
from .connector import WhatsappConnector


def main() -> None:
    asyncio.run(WhatsappConnector(Settings()).run_until_stopped())


if __name__ == "__main__":
    main()
