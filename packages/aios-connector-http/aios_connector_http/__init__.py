"""Runtime-container SDK for aios connectors.

One container hosts N connections of one ``connector`` *type*; the
:class:`HttpConnector` base class handles discovery, secrets, tool
dispatch, and inbound emission against the runtime-scoped routes
introduced in #328 PR 5.  Built on the typed :mod:`aios_sdk` workspace
package; clients reach for ``aios_sdk._generated.api.*`` directly when
they need an operation not exposed at the base-class level.
"""

from __future__ import annotations

from .runner import (
    HttpConnector,
    ManagementHandlerError,
    SandboxPathError,
    management_handler,
    tool,
)
from .sandbox import Attachment, AttachmentError, SandboxPath
from .spool import SqliteAnsweredSpool

__all__ = [
    "Attachment",
    "AttachmentError",
    "HttpConnector",
    "ManagementHandlerError",
    "SandboxPath",
    "SandboxPathError",
    "SqliteAnsweredSpool",
    "management_handler",
    "tool",
]
