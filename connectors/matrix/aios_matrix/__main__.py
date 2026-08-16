from __future__ import annotations

import asyncio

from .connector import MatrixConnector

if __name__ == "__main__":
    asyncio.run(MatrixConnector().run())
