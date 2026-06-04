"""Unit tests for the opaque pagination cursor codec (src/aios/models/pagination.py)."""

from __future__ import annotations

import base64
import json

import pytest

from aios.errors import ValidationError
from aios.models.pagination import (
    CursorState,
    decode_cursor,
    encode_cursor,
    reject_params_with_cursor,
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
