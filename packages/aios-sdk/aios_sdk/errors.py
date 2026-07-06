"""The SDK's single error-envelope decoder + raw-body request helper.

The aios server returns every error as an envelope::

    {"error": {"type": "...", "message": "...", "detail": {...}}}

This module owns the *one* decoder for that shape plus the shared
:class:`AiosApiError` the CLI's ``run_or_die`` shim catches. It lives in
``aios_sdk`` proper (NOT ``_generated/`` — the codegen drift-guard
overwrites that tree) and, crucially, decodes the envelope by hand rather
than importing ``aios.models.common.ErrorResponse``: the SDK is a
standalone workspace package and must not import back into ``aios`` (a
layering cycle CI wouldn't catch).

:func:`raw_request` is the deliberate thin-wire arm — the CLI's
``--file``/``--stdin``/``--data`` and flag-built payloads ride raw dicts
straight to the server so an older CLI never client-side-rejects a
payload a newer server accepts. Codegen can't express that intent, so it
lives here beside :mod:`aios_sdk.streaming`, the established home for what
the generated client can't model.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from aios_sdk._generated import AuthenticatedClient


class AiosApiError(Exception):
    """Raised when the API returns a non-2xx response (or the transport fails).

    Carries the server's error envelope shape when available. A transport
    failure (connection refused, timeout) is represented as ``status_code
    == 0`` with a synthetic ``error_type`` so the CLI renders a clean
    message instead of a raw ``httpx`` traceback.
    """

    def __init__(
        self,
        *,
        status_code: int,
        error_type: str,
        message: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{error_type} ({status_code}): {message}")
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.detail = detail or {}


def error_from_response(status_code: int, content: bytes | str) -> AiosApiError:
    """Decode a non-2xx response body into an :class:`AiosApiError`.

    The single implementation of the server's error-envelope contract. A
    body shaped ``{"error": {"type", "message", "detail"}}`` decodes to the
    server's own type/message; anything else falls back to ``http_error``
    with the raw text so the user still sees *something* actionable.
    """
    text = content.decode(errors="replace") if isinstance(content, bytes) else content
    try:
        body = json.loads(text) if text else None
    except (ValueError, json.JSONDecodeError):
        body = None
    if not isinstance(body, dict):
        return AiosApiError(
            status_code=status_code,
            error_type="http_error",
            message=text.strip() or f"HTTP {status_code}",
        )
    error = body.get("error")
    if not isinstance(error, dict) or "type" not in error or "message" not in error:
        return AiosApiError(
            status_code=status_code,
            error_type="http_error",
            message=json.dumps(body),
        )
    detail = error.get("detail")
    return AiosApiError(
        status_code=status_code,
        error_type=str(error["type"]),
        message=str(error["message"]),
        detail=detail if isinstance(detail, dict) else None,
    )


def raise_for_response(response: httpx.Response) -> None:
    """Raise :class:`AiosApiError` if ``response`` is non-2xx (reads the body)."""
    if 200 <= response.status_code < 300:
        return
    raise error_from_response(int(response.status_code), response.content)


def _prune(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if params is None:
        return None
    return {k: v for k, v in params.items() if v is not None}


def raw_request(
    client: AuthenticatedClient,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Send a raw-dict request over the SDK client's ``httpx.Client``.

    The deliberate thin-wire arm: the body passes through untyped so the
    server stays the single validator of schema-fluid payloads (sessions
    create/update/messages/triggers/resources). Returns the decoded JSON
    body, or ``None`` for a 204 / empty 2xx. Non-2xx raises
    :class:`AiosApiError`; transport failures map to ``status_code == 0``
    so the CLI never leaks an ``httpx`` traceback.

    Auth + base URL come from ``client`` (its underlying ``httpx.Client``
    already carries the Bearer token), so the raw arm and every generated
    op share one transport, one connection pool, and one auth header.
    """
    httpx_client = client.get_httpx_client()
    try:
        response = httpx_client.request(method, path, json=json_body, params=_prune(params))
    except httpx.ConnectError as exc:
        raise AiosApiError(
            status_code=0,
            error_type="connection_error",
            message=f"could not connect to {httpx_client.base_url}: {exc}",
        ) from exc
    except httpx.TimeoutException as exc:
        raise AiosApiError(
            status_code=0,
            error_type="timeout",
            message=f"request to {path} timed out: {exc}",
        ) from exc
    if 200 <= response.status_code < 300:
        if response.status_code == 204 or not response.content:
            return None
        return response.json()
    raise error_from_response(int(response.status_code), response.content)
