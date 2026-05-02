"""Stdio entry point for the echo connector test fixture."""

from __future__ import annotations

import anyio
from tests.fixtures.echo_connector.server import run


def main() -> None:
    anyio.run(run)


if __name__ == "__main__":
    main()
