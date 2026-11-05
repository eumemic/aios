"""Runtime-container SDK for aios connectors.

One container hosts N connections of one ``connector`` *type*; the
:class:`HttpConnector` base class handles discovery, secrets, tool
dispatch, and inbound emission against the runtime-scoped routes
introduced in #328 PR 5.  Built on the typed :mod:`aios_sdk` workspace
package; clients reach for ``aios_sdk._generated.api.*`` directly when
they need an operation not exposed at the base-class level.
"""

from __future__ import annotations

from ._time import iso_from_ms
from .runner import (
    HttpConnector,
    ManagementHandlerError,
    SandboxPathError,
    management_handler,
    tool,
)
from .sandbox import (
    FALLBACK_MARKER,
    MAX_ATTACHMENT_BYTES,
    Attachment,
    AttachmentError,
    SandboxPath,
    append_markers,
    coerce_size,
    drop_attachment,
    format_size,
    is_failed_download,
    is_over_cap,
    rejection_marker,
    safe_marker,
)
from .spool import SqliteAnsweredSpool

__all__ = [
    "FALLBACK_MARKER",
    "MAX_ATTACHMENT_BYTES",
    "Attachment",
    "AttachmentError",
    "HttpConnector",
    "ManagementHandlerError",
    "SandboxPath",
    "SandboxPathError",
    "SqliteAnsweredSpool",
    "append_markers",
    "coerce_size",
    "drop_attachment",
    "format_size",
    "is_failed_download",
    "is_over_cap",
    "iso_from_ms",
    "management_handler",
    "rejection_marker",
    "safe_marker",
    "tool",
]
