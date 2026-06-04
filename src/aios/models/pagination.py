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
from typing import Any, Literal

from aios.errors import ValidationError

Direction = Literal["forward", "backward"]

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
