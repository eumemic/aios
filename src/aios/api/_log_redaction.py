"""Leaf-module home for the request-log path redactor.

Imports only the stdlib ``re`` — deliberately NOTHING from ``aios`` — so it can
be imported from both the API layer (``aios.api.middleware``) and the
foundational ``aios.errors`` (imported by ~39 db/harness/services/workflows
modules) without dragging the starlette/API layer into that foundation or
creating an import cycle.
"""

from __future__ import annotations

import re

_INGEST_PATH_RE = re.compile(r"^(/v1/triggers/ingest/)[^/]+")


def redact_sensitive_path(path: str) -> str:
    """Replace the per-trigger ingest bearer token in the URL path with a
    placeholder so request/error logs never persist a live credential.

    Only the one known secret-bearing route (``POST
    /v1/triggers/ingest/{ingest_token}`` — see
    ``aios.api.routers.triggers_ingest``) is rewritten; all other paths pass
    through unchanged (a broad heuristic would risk mangling legitimate ids).
    """
    return _INGEST_PATH_RE.sub(r"\1<redacted>", path)
