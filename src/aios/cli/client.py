"""Synchronous HTTP client for the aios API.

A thin wrapper around :class:`httpx.Client` that handles:

* Bearer auth header injection.
* Error-envelope decoding (server returns ``{"error": {"type","message","detail"}}``).
* SSE streaming via :meth:`stream_session`.

Resource methods accept dicts so the CLI does not re-implement pydantic
models client-side — the server's validators remain the single source of
truth. The only client-side shape the CLI applies is the list envelope
(``ListResponse[T]``) which is stable and documented.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from pydantic import ValidationError

from aios.cli.sse import SseMessage, parse_sse_lines
from aios.models.common import ErrorResponse


class AiosApiError(Exception):
    """Raised when the API returns a non-2xx response.

    Carries the server's error envelope shape when available.
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


class AiosClient:
    """Synchronous client bound to a single aios API base URL + bearer key."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, read=timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AiosClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── low-level request helper ─────────────────────────────────────────

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Send a request and return the decoded JSON body.

        Returns ``None`` for 204 responses. Raises :class:`AiosApiError`
        on non-2xx.
        """
        try:
            response = self._client.request(
                method,
                path,
                json=json_body,
                params=_prune_params(params),
            )
        except httpx.ConnectError as exc:
            raise AiosApiError(
                status_code=0,
                error_type="connection_error",
                message=f"could not connect to {self.base_url}: {exc}",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AiosApiError(
                status_code=0,
                error_type="timeout",
                message=f"request to {path} timed out: {exc}",
            ) from exc
        if response.status_code == 204 or not response.content:
            if 200 <= response.status_code < 300:
                return None
            _raise_for_error(response)
        if 200 <= response.status_code < 300:
            return response.json()
        _raise_for_error(response)
        return None  # unreachable; keeps type checker happy

    # ── streaming (SSE) ──────────────────────────────────────────────────

    @contextmanager
    def stream_session(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
    ) -> Iterator[Iterator[SseMessage]]:
        """Stream events for a session as :class:`SseMessage` values.

        Used as a context manager to ensure the underlying HTTP connection
        is closed. The yielded iterator is consumed lazily.
        """
        params = _prune_params({"after_seq": after_seq})
        headers = {"Accept": "text/event-stream"}
        try:
            with self._client.stream(
                "GET",
                f"/v1/sessions/{session_id}/stream",
                params=params,
                headers=headers,
                timeout=httpx.Timeout(60.0, read=None),
            ) as response:
                if response.status_code >= 400:
                    # Drain the body so the error envelope is available.
                    response.read()
                    _raise_for_error(response)
                yield parse_sse_lines(response.iter_lines())
        except httpx.ConnectError as exc:
            raise AiosApiError(
                status_code=0,
                error_type="connection_error",
                message=f"could not connect to {self.base_url}: {exc}",
            ) from exc


def _prune_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if params is None:
        return None
    return {k: v for k, v in params.items() if v is not None}


def _raise_for_error(response: httpx.Response) -> None:
    status = response.status_code
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        raise AiosApiError(
            status_code=status,
            error_type="http_error",
            message=response.text.strip() or response.reason_phrase or f"HTTP {status}",
        ) from None

    try:
        envelope = ErrorResponse.model_validate(body)
    except ValidationError:
        raise AiosApiError(
            status_code=status,
            error_type="http_error",
            message=json.dumps(body) if body is not None else f"HTTP {status}",
        ) from None

    raise AiosApiError(
        status_code=status,
        error_type=envelope.error.type,
        message=envelope.error.message,
        detail=envelope.error.detail,
    )
