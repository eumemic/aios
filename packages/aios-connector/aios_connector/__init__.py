"""Reference SDK for building aios connectors.

A connector is a stdio MCP subprocess aios spawns and supervises.  It
exposes:

* **Tools** (``signal_send``, ``telegram_send``, etc.) the model calls.
* **Inbound notifications** (``notifications/aios/inbound``) carrying
  user messages from the underlying platform into aios sessions.
* **Account snapshots** (``notifications/aios/accounts``) the operator
  surfaces in ``aios connector list``.

Subclass :class:`Connector`, decorate methods with :func:`tool`, call
:meth:`Connector.emit_inbound`, and register your factory under the
``aios.connectors`` entry-point group.  See
:doc:`/packages/aios-connector/examples/echo/README` for the canonical
shape.

Public API:

* :class:`Connector` — base class.
* :func:`tool` — decorator publishing a method as an MCP tool.
* :func:`focal_required` — decorator pulling
  ``_meta.aios.focal_channel_path`` into the ``focal`` kwarg.
* :func:`make_account` — convenience builder for account snapshot
  entries.
"""

from __future__ import annotations

from aios_connector.base import (
    Connector,
    focal_required,
    make_account,
    tool,
)

__all__ = ["Connector", "focal_required", "make_account", "tool"]
