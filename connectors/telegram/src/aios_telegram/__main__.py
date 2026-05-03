"""``python -m aios_telegram`` — stdio entry point spawned by aios.

Pre-PR3 this was an argparse-driven CLI with ``start`` subcommand and
flags overriding env vars (operators ran the connector themselves
under systemd / Docker).  Post-PR3 the supervisor spawns this module
directly, so config flows from env vars exclusively (the supervisor
inherits its own env to the child process).
"""

from __future__ import annotations

import sys

import anyio

from .config import Settings
from .connector import TelegramConnector
from .logging import configure_logging


def main() -> int:
    configure_logging()
    cfg = Settings()
    try:
        anyio.run(TelegramConnector(cfg).run)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
