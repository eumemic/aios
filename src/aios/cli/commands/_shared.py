"""Helpers shared by every resource subcommand."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import typer

from aios.cli.output import OutputFormat, print_json, print_note, print_table
from aios.cli.runtime import CliState, get_state
from aios.models.pagination import MAX_PAGE_LIMIT
from aios_sdk import Client, raw_request
from aios_sdk._generated.types import Response, Unset
from aios_sdk.errors import error_from_response


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


def raw_single(
    ctx: typer.Context,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
) -> None:
    """Send a raw-dict request over the SDK client and render the result.

    The thin-wire counterpart to :func:`call_single`: bodies pass through
    untyped (:func:`aios_sdk.raw_request`) so the server stays the sole
    validator of schema-fluid payloads. A 204/empty body prints nothing.
    """
    with get_state(ctx).sdk_client() as client:
        obj = raw_request(client, method, path, json_body=json_body)
    if obj is not None:
        render_single(obj)


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


def unwrap(response: Response[Any]) -> Any:
    """2xx → parsed body (``None`` for 204); non-2xx → :class:`AiosApiError`.

    Delegates envelope decoding to the SDK's single decoder
    (:func:`aios_sdk.error_from_response`) so the wire-error contract has
    exactly one implementation.
    """
    status = int(response.status_code)
    if 200 <= status < 300:
        return response.parsed
    raise error_from_response(status, response.content)


def raw_paginate(
    client: Client,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    page_size: int = MAX_PAGE_LIMIT,
) -> dict[str, Any]:
    """Walk every page of a raw list endpoint and fold into one envelope.

    The thin-wire arm of the *single* page walker (:func:`render_paginated`
    is the typed arm). The first request carries the filters + ``limit``;
    every later request sends only the opaque ``?cursor=`` from the previous
    page's ``next_cursor`` — the server 422s if filters are re-sent
    alongside a cursor. Terminates on ``has_more`` false OR a missing
    cursor.
    """
    accumulated: list[Any] = []
    cursor: str | None = None
    while True:
        if cursor is not None:
            page_params: dict[str, Any] = {"cursor": cursor}
        else:
            page_params = {**(params or {}), "limit": page_size}
        page = raw_request(client, "GET", path, params=page_params)
        assert isinstance(page, dict)
        page_data = page.get("data", [])
        assert isinstance(page_data, list)
        accumulated.extend(page_data)
        cursor = page.get("next_cursor")
        if not page.get("has_more") or cursor is None:
            break
    return {"data": accumulated, "has_more": False, "next_cursor": None}


def render_paginated(
    ctx: typer.Context,
    fn: Callable[..., Response[Any]],
    *,
    columns: Sequence[str],
    all_: bool,
    limit: int = 50,
    max_widths: dict[str, int] | None = None,
    page_size: int = MAX_PAGE_LIMIT,
    path_params: dict[str, Any] | None = None,
    **filters: Any,
) -> None:
    """Render an SDK list endpoint, paginated or fully fetched per ``all_``.

    Single entry point used by every ``aios <resource> list`` command:
    opens an SDK client, walks pages when ``all_`` is True (or fetches one
    page otherwise), then renders through :func:`render_list`. The first page
    carries the filters + ``limit``; later pages send only the opaque ``cursor``.

    ``path_params`` (e.g. the ``run_id`` of ``aios runs events``) are part of the
    URL, so they are sent on EVERY page — unlike ``filters`` (query params), which
    the opaque cursor carries and the server rejects (422) if re-sent alongside
    ``?cursor=``. Without this, a ``--all`` walk of a path-param'd endpoint dropped
    the path param on page 2 and the SDK call raised on the missing argument.
    """
    path = path_params or {}
    state = get_state(ctx)
    with state.sdk_client() as client:
        if all_:
            items: list[Any] = []
            cursor: str | None = None
            while True:
                if cursor is not None:
                    page = unwrap(fn(client=client, cursor=cursor, **path))
                else:
                    page = unwrap(fn(client=client, limit=page_size, **path, **filters))
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
            page = unwrap(fn(client=client, limit=limit, **path, **filters))
            envelope = page.to_dict()
    render_list(state.output_format, envelope, columns=columns, max_widths=max_widths)


def raw_paginate_events(
    client: Client,
    session_id: str,
    *,
    kind: str | None = None,
    direction: str = "forward",
    page_size: int = MAX_PAGE_LIMIT,
) -> list[dict[str, Any]]:
    """Walk every page of ``/v1/sessions/:id/events`` and return the raw list.

    A variant of :func:`raw_paginate` that preserves the events endpoint's
    distinct semantics: it breaks on an *empty page* (not just ``has_more``
    false), and the first request carries ``dir``/``kind``/``limit`` while
    later requests send only the opaque ``?cursor=``. Returns just the event
    list; callers wrap it in an envelope for ``render_list`` or consume it
    directly (the profiler).
    """
    accumulated: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        if cursor is not None:
            params: dict[str, Any] = {"cursor": cursor}
        else:
            params = {"dir": direction, "kind": kind, "limit": page_size}
        page = raw_request(client, "GET", f"/v1/sessions/{session_id}/events", params=params)
        assert isinstance(page, dict)
        page_data = page.get("data", [])
        if not page_data:
            break
        accumulated.extend(page_data)
        cursor = page.get("next_cursor")
        if not page.get("has_more") or cursor is None:
            break
    return accumulated


def get_state_and_client(ctx: typer.Context) -> tuple[CliState, Client]:
    """Return (state, sdk_client) for a raw list/events command body.

    The single accessor for commands that need ``output_format``/``verbose``
    alongside a raw-arm client. Caller owns the client's ``with`` block.
    """
    state = get_state(ctx)
    return state, state.sdk_client()
