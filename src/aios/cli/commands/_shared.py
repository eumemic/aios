"""Helpers shared by every resource subcommand.

Keeps the per-resource files tiny — they mostly declare columns and call
the helpers here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import typer
from pydantic import ValidationError

from aios.cli.client import AiosApiError, AiosClient
from aios.cli.output import OutputFormat, print_json, print_note, print_table
from aios.cli.runtime import CliState, get_state
from aios.models.common import ErrorResponse
from aios.sdk import Client
from aios.sdk._generated.types import Response, Unset


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


def render_list(
    output_format: OutputFormat,
    envelope: dict[str, Any],
    *,
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    max_widths: dict[str, int] | None = None,
) -> None:
    """Print a ``ListResponse`` envelope honoring ``output_format``.

    JSON: dump the full envelope (data + has_more + next_after).
    Table: render ``data`` rows; emit a dim hint if ``has_more``.
    """
    if output_format == "json":
        print_json(envelope)
        return

    data = envelope.get("data", [])
    assert isinstance(data, list)
    print_table(data, columns, headers=headers, max_widths=max_widths)
    if envelope.get("has_more"):
        nxt = envelope.get("next_after")
        if nxt is not None:
            print_note(f"… more results. Re-run with --after {nxt}")
        else:
            print_note("… more results available")


def fetch_all(
    client: AiosClient,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    page_size: int = 200,
) -> dict[str, Any]:
    """Walk every page of a list endpoint and return one envelope.

    Folded into a single ``{"data": [...], "has_more": false, "next_after": null}``
    envelope so downstream renderers don't care that pagination happened.
    """
    accumulated: list[Any] = []
    cursor: str | None = None
    while True:
        page_params: dict[str, Any] = dict(params or {})
        page_params["limit"] = page_size
        if cursor is not None:
            page_params["after"] = cursor
        page = client.request("GET", path, params=page_params)
        assert isinstance(page, dict)
        page_data = page.get("data", [])
        assert isinstance(page_data, list)
        accumulated.extend(page_data)
        if not page.get("has_more"):
            break
        cursor = page.get("next_after")
        if cursor is None:
            break
    return {"data": accumulated, "has_more": False, "next_after": None}


class _ListPage(Protocol):
    """Structural shape of every generated ``ListResponse<T>`` model."""

    data: list[Any]
    has_more: bool | Unset
    next_after: None | str | Unset


def unwrap(response: Response[Any]) -> Any:
    """Translate an SDK ``Response`` into its parsed body or :class:`AiosApiError`.

    Mirrors :meth:`AiosClient.request`: 2xx returns the parsed body (or
    ``None`` for 204); non-2xx is decoded against the server's
    ``ErrorResponse`` envelope and re-raised so ``run_or_die`` can render
    the same friendly message regardless of whether the call went through
    the hand-written client or the typed SDK.

    Returns ``Any`` rather than the caller's ``T`` because the SDK's
    ``Response[T]`` runs through ``attrs.define + Generic[T]``, a combo
    mypy widens to ``Any`` anyway — preserving the generic in the helper
    signature would be theatre. Callers that want the typed view assign
    the result through a typed local (e.g. ``page: ListResponseAgent =
    unwrap(...)``).
    """
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


def fetch_all_sdk(
    fn: Callable[..., Response[Any]],
    *,
    client: Client,
    page_size: int = 200,
    **filters: Any,
) -> list[Any]:
    """Walk every page of an SDK list endpoint. Returns the accumulated typed items.

    Mirrors :func:`fetch_all` for the SDK path: caller hands in the
    operation module's ``sync_detailed``, this walker drives the cursor.
    Filters pass through verbatim, so caller-side typing is preserved.
    """
    accumulated: list[Any] = []
    cursor: str | None = None
    while True:
        kwargs: dict[str, Any] = dict(filters)
        kwargs["limit"] = page_size
        if cursor is not None:
            kwargs["after"] = cursor
        page: _ListPage = unwrap(fn(client=client, **kwargs))
        accumulated.extend(page.data)
        if isinstance(page.has_more, Unset) or not page.has_more:
            break
        if isinstance(page.next_after, Unset) or page.next_after is None:
            break
        cursor = page.next_after
    return accumulated


def render_sdk_list(
    output_format: OutputFormat,
    items: list[Any],
    *,
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    max_widths: dict[str, int] | None = None,
    has_more: bool = False,
    next_after: str | None = None,
) -> None:
    """Render a list of typed SDK models. Models are dict-converted at the boundary.

    The renderer below already speaks dicts; this is the join point where
    SDK types lose their attribute-access affordance — acceptable since the
    CLI only displays the values, never reads specific fields.
    """
    rows = [item.to_dict() for item in items]
    envelope = {"data": rows, "has_more": has_more, "next_after": next_after}
    render_list(output_format, envelope, columns=columns, headers=headers, max_widths=max_widths)


def fetch_all_events(
    client: AiosClient,
    session_id: str,
    *,
    kind: str | None = None,
    after_seq: int = 0,
    page_size: int = 200,
) -> list[dict[str, Any]]:
    """Walk every page of ``/v1/sessions/:id/events`` and return the raw list.

    The events endpoint paginates by ``after_seq`` (monotonic session seq),
    not by the cursor id that :func:`fetch_all` consumes — so it needs its
    own walker. Returns just the event list; callers can wrap in an
    envelope for ``render_list`` or consume directly.
    """
    accumulated: list[dict[str, Any]] = []
    cursor_seq = after_seq
    while True:
        page = client.request(
            "GET",
            f"/v1/sessions/{session_id}/events",
            params={"after_seq": cursor_seq, "kind": kind, "limit": page_size},
        )
        assert isinstance(page, dict)
        page_data = page.get("data", [])
        if not page_data:
            break
        accumulated.extend(page_data)
        last_seq = page_data[-1].get("seq")
        if not page.get("has_more") or last_seq is None:
            break
        cursor_seq = int(last_seq)
    return accumulated
