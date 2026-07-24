from __future__ import annotations

import asyncio

from .appservice import create_appservice
from .config import MatrixConfig


async def main() -> None:
    config = MatrixConfig()
    host, port = config.listen
    appservice = create_appservice(config)
    await appservice.start(host, port)
    try:
        await asyncio.Event().wait()
    finally:
        await appservice.stop()


if __name__ == "__main__":
    asyncio.run(main())
