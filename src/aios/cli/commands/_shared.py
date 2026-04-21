"""Helpers shared by every resource subcommand.

Keeps the per-resource files tiny — they mostly declare columns and call
the helpers here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import typer

from aios.cli.client import AiosClient
from aios.cli.output import print_json, print_table
from aios.cli.runtime import CliState, get_state


def with_client(ctx: typer.Context) -> tuple[CliState, AiosClient]:
    """Return (state, client) for a command body."""
    state = get_state(ctx)
    return state, state.client()


def render_single(state: CliState, obj: Any) -> None:
    """Print a single resource. Always JSON regardless of --format (matches the
    server's shape directly), because single-resource dicts contain nested
    structures that don't tabulate well.
    """
    print_json(obj)


def render_list(
    state: CliState,
    envelope: dict[str, Any],
    *,
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    max_widths: dict[str, int] | None = None,
) -> None:
    """Print a ListResponse envelope.

    In ``table`` format, prints a table of ``data`` rows using ``columns``,
    and emits a dim hint if ``has_more`` is True. In ``json`` format,
    dumps the full envelope (data + has_more + next_after).
    """
    if state.output_format == "json":
        print_json(envelope)
        return

    data = envelope.get("data", [])
    assert isinstance(data, list)
    print_table(data, columns, headers=headers, max_widths=max_widths)
    if envelope.get("has_more"):
        nxt = envelope.get("next_after")
        from aios.cli.output import print_note

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
