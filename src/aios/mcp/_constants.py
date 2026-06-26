"""Shared low-level constants for the MCP transport.

A tiny dependency-light home for values that both :mod:`aios.mcp.client` and
:mod:`aios.mcp.pool` need. ``client`` already imports from ``pool``
(``HttpErrorSink``), so the timeout can't live in ``client`` and be imported
into ``pool`` without an import cycle — this module breaks that by being the
single source of truth both import.
"""

from __future__ import annotations

import httpx

# httpx client bounds for MCP transport. ``read`` is the longest leg — tool
# calls that do real work (DB lookups, external APIs) commonly take tens of
# seconds. Connect/write/pool are tight because they're network fast paths.
# Shared by the per-call client transport (:mod:`aios.mcp.client`) and the
# pool's pooled sessions (:mod:`aios.mcp.pool`) so a stalled MCP server can't
# keep the worker on a dead socket indefinitely.
_MCP_HTTPX_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
