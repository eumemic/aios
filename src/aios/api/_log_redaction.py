"""Leaf-module home for the request-log path redactor.

Imports only the stdlib ``re`` — deliberately NOTHING from ``aios`` — so it can
be imported from both the API layer (``aios.api.middleware``) and the
foundational ``aios.errors`` (imported by ~39 db/harness/services/workflows
modules) without dragging the starlette/API layer into that foundation or
creating an import cycle.
"""

from __future__ import annotations

import re

_INGEST_PATH_RE = re.compile(r"^/*v1/triggers/ingest/+[^/]+", re.IGNORECASE)


def redact_sensitive_path(path: str) -> str:
    """Replace the per-trigger ingest bearer token in the URL path with a
    placeholder so request/error logs never persist a live credential.

    Only the one known secret-bearing route (``POST
    /v1/triggers/ingest/{ingest_token}`` — see
    ``aios.api.routers.triggers_ingest``) is rewritten; all other paths pass
    through unchanged (a broad heuristic would risk mangling legitimate ids).

    The prefix match tolerates duplicated leading slashes and extra separators
    before the token, and is case-insensitive. uvicorn does not normalize
    ``scope["path"]``, so an ingest-shaped path that carries a live token must
    still be redacted regardless of those variations — e.g. ``//v1/...`` (a
    trailing-slash base-URL join), ``/v1/triggers/ingest//<token>`` (an extra
    separator before the token), or a ``/V1/...`` case variant. The replacement
    always yields the canonical single-slash redacted form
    ``/v1/triggers/ingest/<redacted>`` regardless of how many slashes the
    inbound path carried.
    """
    return _INGEST_PATH_RE.sub(r"/v1/triggers/ingest/<redacted>", path)
