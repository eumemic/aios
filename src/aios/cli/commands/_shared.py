"""Helpers shared by every resource subcommand."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import typer
from pydantic import ValidationError

from aios.cli.client import AiosApiError, AiosClient
from aios.cli.output import OutputFormat, print_json, print_note, print_table
from aios.cli.runtime import CliState, get_state
from aios.models.common import ErrorResponse
from aios_sdk._generated.types import Response, Unset


def with_client(ctx: typer.Context) -> tuple[CliState, AiosClient]:
    """Return (state, client) for a command body. Use when ``output_format`` or
    ``verbose`` is needed alongside the client."""
    state = get_state(ctx)
    return state, state.client()


def just_client(ctx: typer.Context) -> AiosClient:
    """Return only the client. Use for commands that never render a list."""
    return get_state(ctx).client()


def render_single(obj: Any) -> None:
    """Print a single resource. Always JSON regardless of ``--format`` because
    single-resource dicts contain nested structures that don't tabulate well.
    """
    print_json(obj)


def call_single(
    ctx: typer.Context,
    fn: Callable[..., Response[Any]],
    **kwargs: Any,
) -> None:
    """Open an SDK client, call an endpoint returning a single resource,
    and render the result through :func:`render_single`.

    The single-resource counterpart to :func:`render_paginated`: every
    ``get`` / ``create`` / ``update`` / ``archive`` body that ends in
    ``render_single(obj.to_dict())`` collapses to one line.
    """
    with get_state(ctx).sdk_client() as client:
        obj = unwrap(fn(client=client, **kwargs))
    render_single(obj.to_dict())


def render_list(
    output_format: OutputFormat,
    envelope: dict[str, Any],
    *,
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    max_widths: dict[str, int] | None = None,
) -> None:
    """Print a ``ListResponse`` envelope honoring ``output_format``.

    JSON: dump the full envelope (data + has_more + next_cursor).
    Table: render ``data`` rows; emit a dim hint if ``has_more``.
    """
    if output_format == "json":
        print_json(envelope)
        return

    data = envelope.get("data", [])
    assert isinstance(data, list)
    print_table(data, columns, headers=headers, max_widths=max_widths)
    if envelope.get("has_more"):
        # Cursor tokens aren't human-authorable, so we point at --all rather
        # than echoing the opaque next_cursor for the user to paste back.
        print_note("… more results — pass --all to fetch every page")


def fetch_all(
    client: AiosClient,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    page_size: int = 200,
) -> dict[str, Any]:
    """Walk every page of a list endpoint and return one envelope.

    The first request carries the filters + ``limit``; every later request sends
    only the opaque ``?cursor=`` from the previous page's ``next_cursor``.
    Folded into a single ``{"data": [...], "has_more": false, "next_cursor": null}``
    envelope so downstream renderers don't care that pagination happened.
    """
    accumulated: list[Any] = []
    cursor: str | None = None
    while True:
        if cursor is not None:
            page_params: dict[str, Any] = {"cursor": cursor}
        else:
            page_params = {**(params or {}), "limit": page_size}
        page = client.request("GET", path, params=page_params)
        assert isinstance(page, dict)
        page_data = page.get("data", [])
        assert isinstance(page_data, list)
        accumulated.extend(page_data)
        cursor = page.get("next_cursor")
        if not page.get("has_more") or cursor is None:
            break
    return {"data": accumulated, "has_more": False, "next_cursor": None}


def unwrap(response: Response[Any]) -> Any:
    """2xx → parsed body (``None`` for 204); non-2xx → :class:`AiosApiError`."""
    status = int(response.status_code)
    if 200 <= status < 300:
        return response.parsed
    try:
        body = json.loads(response.content) if response.content else None
    except (ValueError, json.JSONDecodeError):
        body = None
    if body is None:
        raise AiosApiError(
            status_code=status,
            error_type="http_error",
            message=response.content.decode(errors="replace") or f"HTTP {status}",
        )
    try:
        envelope = ErrorResponse.model_validate(body)
    except ValidationError:
        raise AiosApiError(
            status_code=status, error_type="http_error", message=json.dumps(body)
        ) from None
    raise AiosApiError(
        status_code=status,
        error_type=envelope.error.type,
        message=envelope.error.message,
        detail=envelope.error.detail,
    )


def render_paginated(
    ctx: typer.Context,
    fn: Callable[..., Response[Any]],
    *,
    columns: Sequence[str],
    all_: bool,
    limit: int = 50,
    max_widths: dict[str, int] | None = None,
    page_size: int = 200,
    **filters: Any,
) -> None:
    """Render an SDK list endpoint, paginated or fully fetched per ``all_``.

    Single entry point used by every ``aios <resource> list`` command:
    opens an SDK client, walks pages when ``all_`` is True (or fetches one
    page otherwise), then renders through :func:`render_list`. The first page
    carries the filters + ``limit``; later pages send only the opaque ``cursor``.
    """
    state = get_state(ctx)
    with state.sdk_client() as client:
        if all_:
            items: list[Any] = []
            cursor: str | None = None
            while True:
                if cursor is not None:
                    page = unwrap(fn(client=client, cursor=cursor))
                else:
                    page = unwrap(fn(client=client, limit=page_size, **filters))
                items.extend(page.data)
                if isinstance(page.has_more, Unset) or not page.has_more:
                    break
                if isinstance(page.next_cursor, Unset) or page.next_cursor is None:
                    break
                cursor = page.next_cursor
            envelope: dict[str, Any] = {
                "data": [item.to_dict() for item in items],
                "has_more": False,
                "next_cursor": None,
            }
        else:
            page = unwrap(fn(client=client, limit=limit, **filters))
            envelope = page.to_dict()
    render_list(state.output_format, envelope, columns=columns, max_widths=max_widths)


def fetch_all_events(
    client: AiosClient,
    session_id: str,
    *,
    kind: str | None = None,
    direction: str = "forward",
    page_size: int = 200,
) -> list[dict[str, Any]]:
    """Walk every page of ``/v1/sessions/:id/events`` and return the raw list.

    The first request carries ``dir``/``kind``/``limit``; every later request
    sends only the opaque ``?cursor=`` from the previous page's ``next_cursor``.
    Returns just the event list; callers can wrap it in an envelope for
    ``render_list`` or consume it directly.
    """
    accumulated: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        if cursor is not None:
            params: dict[str, Any] = {"cursor": cursor}
        else:
            params = {"dir": direction, "kind": kind, "limit": page_size}
        page = client.request("GET", f"/v1/sessions/{session_id}/events", params=params)
        assert isinstance(page, dict)
        page_data = page.get("data", [])
        if not page_data:
            break
        accumulated.extend(page_data)
        cursor = page.get("next_cursor")
        if not page.get("has_more") or cursor is None:
            break
    return accumulated
