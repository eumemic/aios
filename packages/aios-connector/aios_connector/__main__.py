"""``python -m aios_connector run <connector_name>`` — connector subprocess entry point.

Spawned by :class:`aios.harness.connector_supervisor.ConnectorSubprocessRegistry`
for every entry in ``settings.connectors_enabled``.  Looks up the
``aios.connectors`` entry point, instantiates the connector, configures
logging, and parks on :meth:`Connector.run` until stdin EOF.

Connector authors do not invoke this themselves; the supervisor builds
the spec internally.  The ``run`` subcommand exists so a future
``debug``/``inspect`` sibling can land without breaking the wire format.
"""

from __future__ import annotations

import sys
from importlib.metadata import entry_points

import anyio

from aios_connector.logging import configure_logging


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        print("usage: python -m aios_connector run <connector_name>", file=sys.stderr)
        return 2
    name = sys.argv[2]
    matches = [ep for ep in entry_points(group="aios.connectors") if ep.name == name]
    if not matches:
        print(f"no aios.connectors entry point named {name!r}", file=sys.stderr)
        return 1
    connector = matches[0].load()()
    configure_logging(connector_name=connector.name)
    try:
        anyio.run(connector.run)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
