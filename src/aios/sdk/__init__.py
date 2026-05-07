"""Typed Python SDK for the aios management plane.

The bulk of this package is auto-generated from ``openapi.json`` via
``scripts/regen-client.sh`` and lives at :mod:`aios.sdk._generated`. This
module re-exports the curated public surface; advanced callers wanting
a specific operation module reach into ``aios.sdk._generated.api.<tag>.<op>``.

Typical usage::

    from aios.sdk import Client, client_from_env, stream_session
    from aios.sdk._generated.api.agents import list_agents

    client = client_from_env()
    response = list_agents.sync_detailed(client=client)

    with stream_session(client, session_id) as events:
        for msg in events:
            print(msg.event, msg.data)
"""

from __future__ import annotations

from aios.sdk._generated import AuthenticatedClient as Client
from aios.sdk._generated.errors import UnexpectedStatus
from aios.sdk.factory import client_from_env
from aios.sdk.streaming import SseMessage, parse_sse_lines, stream_session

__all__ = [
    "Client",
    "SseMessage",
    "UnexpectedStatus",
    "client_from_env",
    "parse_sse_lines",
    "stream_session",
]
