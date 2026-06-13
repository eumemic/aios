"""Foreclosure test for the single-sourced reaction watermark (#1080).

The reaction watermark — ``MAX(COALESCE(reacting_to, seq))`` over assistant
messages — must live in exactly **one** writable place: the ``last_reacted_seq``
``UPDATE`` in ``append_event`` (seeded once by migration 0066's backfill).

Before #1080, ``UNREACTED_ROWS_SQL`` **recomputed** that same watermark via a
duplicate ``session_max_reacting`` CTE (the literal partial migration left by
#750). Any change to reaction semantics had to land in three coordinated places
with nothing forcing the sweep CTE to track — the #155-class drift hazard.

These assertions fail the moment someone reconstitutes the split by authoring a
fresh watermark formula at the sweep site instead of joining the maintained
scalar. They run without Postgres: they inspect the query text directly.
"""

from __future__ import annotations

import re

from aios.harness.sweep import UNREACTED_ROWS_SQL


def test_unreacted_rows_does_not_redefine_the_watermark() -> None:
    """The duplicate ``session_max_reacting`` CTE — and the watermark formula it
    re-derived — must not exist at the sweep site. Reconstituting the split
    requires deliberately re-authoring this aggregate, which this catches."""
    sql = UNREACTED_ROWS_SQL.lower()

    assert "session_max_reacting" not in sql, (
        "UNREACTED_ROWS_SQL re-introduced the session_max_reacting CTE. The "
        "reaction watermark must be single-sourced from sessions.last_reacted_seq "
        "(maintained in append_event), not recomputed here (#1080)."
    )

    # The watermark formula is MAX(COALESCE(... 'reacting_to' ..., seq)). Catch
    # any re-derivation of it regardless of whitespace/alias choices.
    redefines_watermark = re.search(r"max\s*\(\s*coalesce\s*\(.*reacting_to", sql, re.DOTALL)
    assert redefines_watermark is None, (
        "UNREACTED_ROWS_SQL recomputes the reaction watermark "
        "(MAX(COALESCE(reacting_to, seq))). This formula must live in exactly one "
        "writable place — append_event's last_reacted_seq UPDATE (#1080)."
    )


def test_unreacted_rows_joins_the_maintained_scalar() -> None:
    """The unreacted-rows gate must consume the maintained scalar directly:
    JOIN sessions and filter ``e.seq > s.last_reacted_seq``."""
    sql = UNREACTED_ROWS_SQL.lower()

    assert "join sessions" in sql, (
        "UNREACTED_ROWS_SQL must JOIN sessions to read the maintained "
        "last_reacted_seq watermark (#1080)."
    )
    assert "last_reacted_seq" in sql, (
        "UNREACTED_ROWS_SQL must filter on the maintained last_reacted_seq scalar (#1080)."
    )
    # The filter compares the candidate event seq against the maintained scalar.
    assert re.search(r"e\.seq\s*>\s*s\.last_reacted_seq", sql), (
        "UNREACTED_ROWS_SQL must gate on e.seq > s.last_reacted_seq (#1080)."
    )


def test_unreacted_rows_still_excludes_assistant_messages() -> None:
    """Behavior preservation: the gate still returns only non-assistant
    ('user' + 'tool') message rows."""
    sql = UNREACTED_ROWS_SQL.lower()
    assert "role <> 'assistant'" in sql or "role != 'assistant'" in sql, (
        "UNREACTED_ROWS_SQL must still exclude assistant messages."
    )
    assert "kind = 'message'" in sql
