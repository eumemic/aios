"""Opaque cursor tokens for list pagination.

Every paginated list endpoint returns its next-page cursor as an opaque
``next_cursor`` token. The first page is requested with filters + ``limit``
(+ ``dir`` for the events endpoint); every subsequent page is just
``?cursor=<token>``. The token is self-contained — it carries the keyset
position, the direction, the active filters, and the page size — so a client
walks a result set without re-sending anything.

The token is **unsigned** base64url(JSON). It is an opaque-by-convention
position pointer, not a security boundary: every list query is already scoped
``WHERE account_id = $1``, so a forged or tampered token cannot read across
tenants. Signing would add key management for zero isolation benefit.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from fastapi import Query

from aios.errors import ValidationError

Direction = Literal["forward", "backward"]

# --- The list-endpoint ``limit`` contract -----------------------------------
#
# Single source of truth for the two coupled facts every paginated list
# endpoint declares: the per-page default and the request-boundary ceiling.
# Previously these were 35 bare literals hand-typed across the routers (the
# ``Query(le=N)`` ceiling and the ``… else N`` default), with no named
# constant — so "default exceeds ceiling" was representable and a dropped
# constraint looked identical to a typo.
#
# ``MAX_PAGE_LIMIT`` is also imported by the CLI (``cli/commands/_shared.py``)
# so the cross-process invariant "CLI page_size ≤ server ceiling" is a
# derivation rather than a coincidence two literals happen to satisfy.
#
# The events endpoints legitimately want a higher ceiling (operators page full
# event logs); they pass it as a *named* override (:data:`MAX_EVENT_PAGE_LIMIT`)
# so the deviation is a one-token, greppable, intentional act.
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200
MAX_EVENT_PAGE_LIMIT = 500

# ``PageLimit`` keeps the ``ge``/``le`` literals inline in the request
# signature (FastAPI needs them statically for OpenAPI / SDK regen — a
# computed runtime value would not appear in the schema), while collapsing
# the repeated ``Annotated[int | None, Query(ge=1, le=MAX_PAGE_LIMIT)]`` to a
# single alias. The ceiling still derives from the constant, so the emitted
# ``le`` is byte-identical to the old hand-typed ``200`` and the openapi
# snapshot test is unaffected.
PageLimit = Annotated[int | None, Query(ge=1, le=MAX_PAGE_LIMIT)]

# Events endpoints (session events, run events): same shape, higher ceiling.
EventPageLimit = Annotated[int | None, Query(ge=1, le=MAX_EVENT_PAGE_LIMIT)]

_CURSOR_VERSION = 1
_DIR_TO_WIRE: dict[Direction, str] = {"forward": "f", "backward": "b"}
_WIRE_TO_DIR: dict[str, Direction] = {"f": "forward", "b": "backward"}


@dataclass(frozen=True, slots=True)
class CursorState:
    """The decoded contents of an opaque pagination cursor.

    ``cursor`` is the last returned row's keyset value in its native type — a
    ULID ``str`` for id-keyed endpoints, an ``int`` for version/seq endpoints.
    ``filters`` holds the equality filters that defined the result set (``{}``
    when none). A plain frozen dataclass, not a pydantic model, so the
    ``str | int`` cursor round-trips without union coercion.
    """

    cursor: str | int
    direction: Direction
    limit: int
    filters: dict[str, Any] = field(default_factory=dict)


def encode_cursor(state: CursorState) -> str:
    """Encode a :class:`CursorState` into an opaque base64url token.

    Filter values are always JSON primitives (``str``/``bool``/``None`` — the
    enum-typed filters are ``Literal``, i.e. strings at runtime), so plain
    ``json.dumps`` suffices with no custom encoder.
    """
    payload = {
        "v": _CURSOR_VERSION,
        "c": state.cursor,
        "d": _DIR_TO_WIRE[state.direction],
        "f": state.filters,
        "l": state.limit,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_cursor(token: str) -> CursorState:
    """Decode an opaque token back into a :class:`CursorState`.

    Garbage, truncated, tampered, or wrong-version tokens raise
    :class:`ValidationError` (422). The model recovers by re-issuing the
    original first-page query rather than hand-editing the cursor.
    """
    try:
        padded = token + "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
            raise ValueError("unrecognized cursor version")
        raw_filters = payload["f"]
        return CursorState(
            cursor=payload["c"],
            direction=_WIRE_TO_DIR[payload["d"]],
            limit=int(payload["l"]),
            filters=raw_filters if isinstance(raw_filters, dict) else {},
        )
    except Exception as exc:
        raise ValidationError(
            "Malformed pagination cursor.",
            detail={
                "hint": "Re-issue the original first-page query; cursors are not hand-editable."
            },
        ) from exc


def reject_params_with_cursor(provided: dict[str, Any]) -> None:
    """Raise 422 if any first-page param was supplied alongside ``?cursor=``.

    A cursor is self-contained, so mixing it with filters/limit/direction is
    ambiguous. To change those, the client starts a fresh first-page query.
    """
    unexpected = sorted(key for key, value in provided.items() if value is not None)
    if unexpected:
        raise ValidationError(
            "A '?cursor=' request takes no other pagination or filter params.",
            detail={
                "unexpected_params": unexpected,
                "hint": "To change filters, limit, or direction, start a fresh first-page query.",
            },
        )


def page_cursor(cursor: str | None, first_page_params: dict[str, Any]) -> CursorState | None:
    """Resolve a list request to a decoded cursor, or ``None`` for the first page.

    On a ``?cursor=`` request: reject any first-page param supplied alongside it
    (422), then return the decoded :class:`CursorState`. With no cursor: return
    ``None`` so the router applies its own first-page defaults. Bundling the
    reject-then-decode step here keeps it uniform across all 12 list endpoints —
    no router can decode without first rejecting conflicting params.
    """
    if cursor is None:
        return None
    reject_params_with_cursor(first_page_params)
    return decode_cursor(cursor)


def resolve_page_limit(
    st: CursorState | None,
    limit: int | None,
    *,
    default: int = DEFAULT_PAGE_LIMIT,
) -> int:
    """Resolve the effective page size for a list request.

    Collapses the repeated triple-nested ternary every router hand-wrote::

        page_limit = st.limit if st is not None else (limit if limit is not None else N)

    On a ``?cursor=`` page the size is whatever the cursor carried; on a first
    page it's the supplied ``?limit=`` or ``default``. Endpoints with a
    non-standard default (the events endpoints page 200 at a time) pass it as a
    named ``default=`` override rather than re-typing the literal.
    """
    if st is not None:
        return st.limit
    return limit if limit is not None else default
