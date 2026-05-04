"""Entry-point factory for the echo connector.

The supervisor's ``resolve_connector_specs`` calls this with
``(connector_name, settings)`` and expects a
:class:`aios.mcp.stdio_transport.ConnectorSpec`.  We hand back a spec
that runs ``python -m aios_echo`` so the launch command is self-locating
across editable / wheel / system installs.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from aios.mcp.stdio_transport import ConnectorSpec

if TYPE_CHECKING:
    pass


def make_spec(connector_name: str, settings: Any) -> ConnectorSpec:
    """Build a :class:`ConnectorSpec` launching the echo connector via ``-m``.

    ``cwd`` is left ``None`` so the supervisor stamps
    ``settings.connectors_dir / <name>`` per plan decision #11.
    """
    return ConnectorSpec(
        name=connector_name,
        command=sys.executable,
        args=["-m", "aios_echo"],
    )
