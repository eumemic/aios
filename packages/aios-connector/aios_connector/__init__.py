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
* :class:`Attachment` / :class:`AttachmentError` — inbound binary blobs
  (photos, voice notes, documents) and the SDK-boundary validation
  failure connectors catch.
* :data:`SandboxPath` — type marker for outbound-attachment parameters.
  Annotate ``attachments: list[SandboxPath] | None = None`` (or scalar
  ``SandboxPath``) and the SDK auto-resolves the model-supplied
  in-sandbox path strings to host :class:`pathlib.Path` objects BEFORE
  the tool body runs.  Connector authors don't call any resolver
  themselves; that's the entire point of the marker.
"""

from __future__ import annotations

from aios_connector.base import (
    Attachment,
    AttachmentError,
    Connector,
    focal_required,
    make_account,
    tool,
)
from aios_connector.media import SandboxPath

__all__ = [
    "Attachment",
    "AttachmentError",
    "Connector",
    "SandboxPath",
    "focal_required",
    "make_account",
    "tool",
]
