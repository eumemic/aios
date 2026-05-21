"""``python -m aios_whatsapp`` entry point.

Reads ``AIOS_URL`` and ``AIOS_RUNTIME_TOKEN`` from env (the SDK does
this automatically inside ``HttpConnector.__init__``).  Deployment-shape
fields like ``AIOS_WHATSAPP_DATA_DIR`` feed pydantic-settings; the
phone (the account identity) lives on each connection's encrypted
secrets and is fetched per-connection in ``serve_connection``.
"""

from __future__ import annotations

import asyncio

from .config import Settings
from .connector import WhatsappConnector


def main() -> None:
    asyncio.run(WhatsappConnector(Settings()).run_until_stopped())


if __name__ == "__main__":
    main()
