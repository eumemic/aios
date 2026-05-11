"""``python -m aios_signal`` entry point.

Reads ``AIOS_URL`` and ``AIOS_CONNECTOR_TOKEN`` from env (the SDK does
this automatically inside ``HttpConnector.__init__``).  Deployment-shape
fields like ``AIOS_SIGNAL_CONFIG_DIR`` feed pydantic-settings; the
phone (the account identity) lives on the connection record's
encrypted secrets and is fetched at ``setup()`` time.
"""

from __future__ import annotations

import asyncio

from .config import Settings
from .connector import SignalConnector


def main() -> None:
    asyncio.run(SignalConnector(Settings()).run())


if __name__ == "__main__":
    main()
