"""Regression: bumping ``_CURSOR_VERSION`` invalidates pre-deploy cursors cleanly.

#1939 changed the session-list cursor's keyset payload ``c`` from a bare id
(ULID str) to a JSON ``[timestamp, id]`` keyset. A pre-deploy v=1 cursor would
otherwise ``decode_cursor`` "successfully" and then crash in the router's
``json.loads(c)`` keyset parse (``c`` being a bare ULID, not a JSON array),
surfacing to the client as an opaque 500-ish failure instead of a recoverable
"re-issue the first page" 422.

The fix bumps ``_CURSOR_VERSION`` so ``decode_cursor`` rejects the stale cursor
UP FRONT with the single-sourced malformed-cursor 422. This test reproduces the
stale-cursor rejection and guards against the over-correction (bumping so far,
or breaking encode/decode, that *current* cursors stop round-tripping).
"""

from __future__ import annotations

import base64
import json

import pytest

from aios.errors import ValidationError
from aios.models.pagination import (
    CursorState,
    decode_cursor,
    encode_cursor,
)


def _forge_v1_cursor() -> str:
    """A pre-#1939 session-list cursor: v=1, ``c`` a bare ULID keyset."""
    payload = {
        "v": 1,
        "c": "sess_0123456789",  # OLD id-only keyset, not a [ts, id] JSON array
        "d": "f",
        "f": {"agent_id": None, "status": None, "order_by": "created_at"},
        "l": 50,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


class TestCursorVersionBump:
    def test_stale_v1_cursor_is_rejected_up_front(self) -> None:
        """The stale cursor must be rejected by decode, BEFORE the router's keyset
        parse — otherwise the client gets an opaque crash instead of a 422."""
        with pytest.raises(ValidationError):
            decode_cursor(_forge_v1_cursor())

    def test_current_cursor_still_round_trips(self) -> None:
        """Over-correction guard: bumping the version must NOT break current
        cursors. A freshly-encoded cursor round-trips to an equal state."""
        state = CursorState(
            cursor=json.dumps(["2026-07-12T23:25:00+00:00", "sess_abc"]),
            direction="forward",
            limit=50,
            filters={"order_by": "created_at"},
        )
        assert decode_cursor(encode_cursor(state)) == state

    def test_version_is_no_longer_one(self) -> None:
        """The keyset format changed, so the version constant must have moved off
        1 — otherwise stale and current cursors are indistinguishable."""
        from aios.models.pagination import _CURSOR_VERSION

        assert _CURSOR_VERSION != 1
