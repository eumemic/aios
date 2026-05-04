"""``python -m aios_echo`` entry point — what the supervisor runs."""

from __future__ import annotations

import anyio

from aios_echo.connector import EchoConnector


def main() -> None:
    anyio.run(EchoConnector().run)


if __name__ == "__main__":
    main()
