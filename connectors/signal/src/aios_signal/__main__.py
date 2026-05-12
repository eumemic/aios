"""``python -m aios_signal`` entry point.

Reads ``AIOS_URL`` and ``AIOS_RUNTIME_TOKEN`` from env (the SDK does
this automatically inside ``HttpConnector.__init__``).  Deployment-shape
fields like ``AIOS_SIGNAL_CONFIG_DIR`` feed pydantic-settings; the
phone (the account identity) lives on each connection's encrypted
secrets and is fetched per-connection in ``serve_connection``.
"""

from __future__ import annotations

import asyncio

from .config import Settings
from .connector import SignalConnector


def main() -> None:
    asyncio.run(SignalConnector(Settings()).run())


if __name__ == "__main__":
    main()
