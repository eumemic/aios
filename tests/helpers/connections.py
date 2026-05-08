"""Shared httpx helpers for e2e tests.

E2E tests routinely build ``httpx`` requests with a bearer token —
operator key (``aios_env["AIOS_API_KEY"]``) or connector token from
``POST /v1/connector-tokens``.  Two duplicated patterns were found
across ~17 sites in 13 files:

* **Client construction** — ``httpx.AsyncClient(base_url=...,
  headers={"Authorization": f"Bearer {token}"})`` (sometimes with a
  ``transport=`` kwarg for in-process ``ASGITransport`` tests).
  Use :func:`authed_client`.

* **Per-request auth** — bare ``client.post(url,
  headers={"Authorization": f"Bearer {token}"})`` where the client
  is constructed elsewhere and auth varies per call.  Use
  :func:`bearer`.
"""

from __future__ import annotations

from typing import Any

import httpx


def authed_client(base_url: str, token: str, **kwargs: Any) -> httpx.AsyncClient:
    """Bearer-auth ``httpx.AsyncClient`` for e2e tests.

    Extra kwargs flow through to :class:`httpx.AsyncClient` — handy
    for in-process tests that pass ``transport=ASGITransport(...)``.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
        **kwargs,
    )


def bearer(token: str) -> dict[str, str]:
    """Bearer-auth header dict for per-request use."""
    return {"Authorization": f"Bearer {token}"}
