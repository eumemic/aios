"""Entry-point factory for the telegram connector.

Resolved by the supervisor's ``resolve_connector_specs`` against the
``aios.connectors`` group.  Returns a ``ConnectorSpec`` that runs
``python -m aios_telegram`` so the launch command is interpreter-stable
across editable / wheel / system installs.
"""

from __future__ import annotations

import sys
from typing import Any

from aios.mcp.stdio_transport import ConnectorSpec


def make_spec(connector_name: str, settings: Any) -> ConnectorSpec:
    """Return a :class:`ConnectorSpec` launching the telegram connector via ``-m``.

    ``cwd`` is left ``None`` so the supervisor stamps
    ``settings.connectors_dir / <name>`` per plan decision #11 — the SDK's
    spool and any other per-connector state files land there.
    """
    return ConnectorSpec(
        name=connector_name,
        command=sys.executable,
        args=["-m", "aios_telegram"],
    )
