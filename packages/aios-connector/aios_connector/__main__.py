"""``python -m aios_connector <connector_name>`` — connector subprocess entry point.

Looks up the ``aios.connectors`` entry point, instantiates the
connector, configures logging, and parks on :meth:`Connector.run`
until stdin EOF.
"""

from __future__ import annotations

import sys
from importlib.metadata import entry_points

import anyio

from aios_connector.logging import configure_logging


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m aios_connector <connector_name>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    matches = entry_points(group="aios.connectors", name=name)
    if not matches:
        print(f"no aios.connectors entry point named {name!r}", file=sys.stderr)
        return 1
    connector = next(iter(matches)).load()()
    configure_logging(connector_name=connector.name)
    try:
        anyio.run(connector.run)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
