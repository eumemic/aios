"""Typed Python SDK for the aios management plane.

The bulk of this package is auto-generated from ``openapi.json`` via
``scripts/regen-client.sh`` and lives at :mod:`aios_sdk._generated`. This
module re-exports the curated public surface; advanced callers wanting
a specific operation module reach into ``aios_sdk._generated.api.<tag>.<op>``.

See the package README for usage examples.
"""

from __future__ import annotations

from aios_sdk._generated import AuthenticatedClient as Client
from aios_sdk._generated.errors import UnexpectedStatus
from aios_sdk.authoring import (
    AgentBuilder,
    define_agent,
    define_tool_builtin,
    define_tool_custom,
    define_tool_mcp,
)
from aios_sdk.factory import client_from_env
from aios_sdk.streaming import (
    SseMessage,
    parse_sse_lines,
    stream_connection_discovery,
    stream_connector_calls,
    stream_management_calls,
    stream_run,
    stream_session,
)

__all__ = [
    "AgentBuilder",
    "Client",
    "SseMessage",
    "UnexpectedStatus",
    "client_from_env",
    "define_agent",
    "define_tool_builtin",
    "define_tool_custom",
    "define_tool_mcp",
    "parse_sse_lines",
    "stream_connection_discovery",
    "stream_connector_calls",
    "stream_management_calls",
    "stream_run",
    "stream_session",
]
