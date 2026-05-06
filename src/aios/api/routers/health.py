"""Health check endpoint. Unauthenticated."""

from __future__ import annotations

from fastapi import APIRouter

from aios import __version__

router = APIRouter()


@router.get("/health", operation_id="get_health")
async def health() -> dict[str, str]:
    """Liveness probe. Unauthenticated; returns the running aios version.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{"status": "ok", "version": <version>}`` if the
    process is up.
    """
    return {"status": "ok", "version": __version__}
