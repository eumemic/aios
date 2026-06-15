"""Unit tests for the opaque pagination cursor codec (src/aios/models/pagination.py)."""

from __future__ import annotations

import base64
import json

import pytest

from aios.errors import ValidationError
from aios.models.pagination import (
    MAX_EVENT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    CursorState,
    decode_cursor,
    encode_cursor,
    reject_params_with_cursor,
    resolve_page_limit,
)


class TestRoundTrip:
    def test_str_cursor_round_trips(self) -> None:
        state = CursorState(cursor="01J9XABCDEF", direction="backward", limit=50, filters={})
        decoded = decode_cursor(encode_cursor(state))
        assert decoded == state

    def test_int_cursor_preserves_type(self) -> None:
        # A version/seq cursor must come back as int, not "5".
        decoded = decode_cursor(
            encode_cursor(CursorState(cursor=5, direction="forward", limit=200, filters={}))
        )
        assert decoded.cursor == 5
        assert isinstance(decoded.cursor, int)

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_direction_round_trips(self, direction: str) -> None:
        decoded = decode_cursor(
            encode_cursor(CursorState(cursor=1, direction=direction, limit=10, filters={}))  # type: ignore[arg-type]
        )
        assert decoded.direction == direction

    def test_filters_round_trip(self) -> None:
        state = CursorState(
            cursor="abc",
            direction="backward",
            limit=50,
            filters={"kind": "message", "error_only": True},
        )
        assert decode_cursor(encode_cursor(state)).filters == {
            "kind": "message",
            "error_only": True,
        }

    def test_empty_filters_round_trip(self) -> None:
        decoded = decode_cursor(
            encode_cursor(CursorState(cursor="x", direction="backward", limit=50, filters={}))
        )
        assert decoded.filters == {}

    def test_token_is_url_safe_and_unpadded(self) -> None:
        token = encode_cursor(CursorState(cursor="x", direction="backward", limit=50, filters={}))
        assert "=" not in token and "+" not in token and "/" not in token


class TestDecodeRejectsBadTokens:
    def test_garbage_raises_422(self) -> None:
        with pytest.raises(ValidationError) as exc:
            decode_cursor("not-a-real-token!!!")
        assert exc.value.status_code == 422

    def test_truncated_token_raises(self) -> None:
        token = encode_cursor(
            CursorState(cursor="01J9XABCDEF", direction="backward", limit=50, filters={})
        )
        with pytest.raises(ValidationError):
            decode_cursor(token[: len(token) // 2])

    def test_wrong_version_raises(self) -> None:
        raw = json.dumps({"v": 2, "c": "x", "d": "b", "f": {}, "l": 50}).encode()
        forged = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        with pytest.raises(ValidationError):
            decode_cursor(forged)

    def test_non_dict_payload_raises(self) -> None:
        forged = base64.urlsafe_b64encode(json.dumps([1, 2, 3]).encode()).decode().rstrip("=")
        with pytest.raises(ValidationError):
            decode_cursor(forged)

    def test_unknown_direction_wire_raises(self) -> None:
        raw = json.dumps({"v": 1, "c": "x", "d": "sideways", "f": {}, "l": 50}).encode()
        forged = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        with pytest.raises(ValidationError):
            decode_cursor(forged)


class TestRejectParamsWithCursor:
    def test_raises_when_param_present(self) -> None:
        with pytest.raises(ValidationError) as exc:
            reject_params_with_cursor({"kind": "message", "limit": None, "error_only": None})
        assert exc.value.detail["unexpected_params"] == ["kind"]

    def test_noop_when_all_none(self) -> None:
        reject_params_with_cursor({"kind": None, "limit": None, "error_only": None})  # no raise

    def test_lists_all_offending_params_sorted(self) -> None:
        with pytest.raises(ValidationError) as exc:
            reject_params_with_cursor({"limit": 10, "agent_id": "ag_1", "status": None})
        assert exc.value.detail["unexpected_params"] == ["agent_id", "limit"]


class TestResolvePageLimitBounds:
    """The cursor carries its own page size, but the cursor is unsigned and
    forgeable (see the module docstring). The request-boundary ceiling the
    ``PageLimit``/``EventPageLimit`` Query aliases enforce on ``?limit=`` must
    therefore ALSO be enforced on the cursor-carried value, or a forged token
    drives an out-of-range ``LIMIT`` straight into Postgres: a negative value
    raises an unhandled 500, a huge value is an unbounded-fetch DoS, and zero
    yields a malformed empty page that still reports ``has_more``."""

    def test_normal_cursor_limit_passes_through(self) -> None:
        st = CursorState(cursor="x", direction="forward", limit=50, filters={})
        assert resolve_page_limit(st, None) == 50

    def test_negative_cursor_limit_is_clamped_up(self) -> None:
        st = CursorState(cursor="x", direction="forward", limit=-5, filters={})
        assert resolve_page_limit(st, None) >= 1

    def test_zero_cursor_limit_is_clamped_up(self) -> None:
        st = CursorState(cursor="x", direction="forward", limit=0, filters={})
        assert resolve_page_limit(st, None) >= 1

    def test_huge_cursor_limit_is_clamped_to_ceiling(self) -> None:
        st = CursorState(cursor="x", direction="forward", limit=100_000_000, filters={})
        assert resolve_page_limit(st, None) <= MAX_PAGE_LIMIT

    def test_events_ceiling_not_clamped_to_default_max(self) -> None:
        # A legitimate events cursor carries up to MAX_EVENT_PAGE_LIMIT; the higher
        # ceiling must be honored, not silently shrunk to MAX_PAGE_LIMIT.
        st = CursorState(cursor="x", direction="forward", limit=MAX_EVENT_PAGE_LIMIT, filters={})
        assert resolve_page_limit(st, None, maximum=MAX_EVENT_PAGE_LIMIT) == MAX_EVENT_PAGE_LIMIT

    def test_forged_token_negative_limit_is_bounded_end_to_end(self) -> None:
        # The full attack path: a hand-minted base64url(JSON) token with a negative
        # limit decodes cleanly, then must resolve to a SQL-safe page size.
        raw = json.dumps({"v": 1, "c": "01J9X", "d": "f", "f": {}, "l": -5}).encode()
        forged = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        st = decode_cursor(forged)
        assert resolve_page_limit(st, None) >= 1

    def test_first_page_limit_still_resolves(self) -> None:
        # No cursor: the (already Pydantic-bounded) ?limit= or the default flows through.
        assert resolve_page_limit(None, 25) == 25
        assert resolve_page_limit(None, None) >= 1
