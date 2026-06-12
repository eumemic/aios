"""Structural guard (issue #862): neither the tokenizer pass nor the
parent-assistant channel lookup may execute inside ``append_event``'s row-lock
transaction. Between the seq-allocating UPDATE and the INSERT, the ONLY awaited
query is ``_latest_cumulative_tokens``.

A source-string scan in the spirit of ``TestNoEventsSubquery`` in
``tests/integration/test_session_status_scalars.py`` (which string-scans a SQL
constant) — it needs no DB, so it runs in the fast unit tier.
"""

from __future__ import annotations

import inspect

from aios.db.queries import events as events_mod


def _append_event_source() -> str:
    return inspect.getsource(events_mod.append_event)


class TestNoInLockCompute:
    def test_transaction_opens_after_all_token_and_lookup_calls(self) -> None:
        """The ``async with conn.transaction()`` must appear AFTER the last
        pre-transaction computation (token delta + parent lookup)."""
        src = _append_event_source()
        tx_pos = src.index("async with conn.transaction()")

        # Every compute helper that must run pre-lock appears (if at all)
        # before the transaction opens.
        for needle in (
            "_event_token_delta",
            "_lookup_tool_parent_channel",
            "get_session_focal_channel",
        ):
            if needle in src:
                assert src.index(needle) < tx_pos, (
                    f"{needle} must be called before the transaction opens, not inside the row lock"
                )

    def test_only_cumulative_query_between_update_and_insert(self) -> None:
        """In the slice between the seq-allocating UPDATE and the INSERT, the
        only awaited query helper is ``_latest_cumulative_tokens`` — no
        tokenizer, no channel lookup, no focal read."""
        src = _append_event_source()
        update_pos = src.index('"UPDATE sessions ')
        insert_pos = src.index('"INSERT INTO events ')
        assert update_pos < insert_pos
        between = src[update_pos:insert_pos]

        assert "_latest_cumulative_tokens" in between
        for forbidden in (
            "_lookup_tool_parent_channel",
            "approx_tokens",
            "render_user_event",
            "get_session_focal_channel",
            "_event_token_delta",
        ):
            assert forbidden not in between, (
                f"{forbidden} must not run between the UPDATE and the INSERT "
                "(it belongs pre-transaction)"
            )
