"""``python -m aios_signal`` entry point.

Reads ``AIOS_URL`` and ``AIOS_CONNECTOR_TOKEN`` from env (the SDK does
this automatically inside ``HttpConnector.__init__``).  Signal-specific
config (``AIOS_SIGNAL_PHONE``, ``AIOS_SIGNAL_CONFIG_DIR``) feeds
pydantic-settings.
"""

from __future__ import annotations

import asyncio

from .config import Settings
from .connector import SignalConnector


def main() -> None:
    asyncio.run(SignalConnector(Settings()).run())


if __name__ == "__main__":
    main()
