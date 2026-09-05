"""Write↔read coupling for durable reminder rows (non-stimulus user messages).

A reminder is a ``role='user'`` message row tagged
``metadata[REMINDER_METADATA_KEY]``. ``append_event`` treats it as
non-stimulus (no ``last_stimulus_seq`` / ``last_user_seq`` / ``updated_at``
bump). Every LOG-DERIVED reader of user rows must agree, or a reminder past
the reaction watermark would (a) defeat the sweep's incomplete-batch filter,
(b) consume the inbound budget, (c) be retained by the windower's tail clamp
in place of the real stimulus. These tests pin that every such reader routes
through the single-source constants in ``aios.models.events`` — and that the
SQL form is NULL-safe, since ``data->'metadata'`` is NULL on every row without
a metadata key (tool results, plain user posts) and ``NOT (NULL ? k)`` would
exclude those rows entirely.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from aios.db.queries import events as events_mod
from aios.harness.sweep import UNREACTED_ROWS_FLOORED_SQL, UNREACTED_ROWS_SQL
from aios.models.events import (
    REMINDER_EXCLUDE_SQL,
    REMINDER_METADATA_KEY,
    is_reminder_event,
    reminder_section,
)
from aios.services.inbound import _RESERVED_METADATA_KEYS
from aios.services.inbound_budget import _INFERENCE_BEARING_PREDICATE


def _reminder(section: str = "concise", role: str = "user") -> dict[str, object]:
    return {
        "role": role,
        "content": "reminder text",
        "metadata": {REMINDER_METADATA_KEY: {"section": section, "digest": "d", "v": 1}},
    }


class TestPredicate:
    def test_recognises_a_reminder_row(self) -> None:
        assert reminder_section("message", _reminder("obligations")) == "obligations"
        assert is_reminder_event("message", _reminder()) is True

    def test_rejects_everything_else(self) -> None:
        assert reminder_section("message", {"role": "user", "content": "hi"}) is None
        assert (
            reminder_section("message", {"role": "user", "content": "hi", "metadata": {}}) is None
        )
        assert reminder_section("message", _reminder(role="assistant")) is None
        assert reminder_section("lifecycle", _reminder()) is None
        # An unknown section is NOT trusted as a reminder.
        assert reminder_section("message", _reminder(section="bogus")) is None
        assert is_reminder_event("message", {"role": "tool", "content": "x"}) is False

    def test_marker_shape_is_defensive(self) -> None:
        assert reminder_section("message", {"role": "user", "metadata": "junk"}) is None
        assert (
            reminder_section("message", {"role": "user", "metadata": {REMINDER_METADATA_KEY: 7}})
            is None
        )


class TestSqlForm:
    def test_exclusion_is_null_safe(self) -> None:
        sql = REMINDER_EXCLUDE_SQL.format(col="e.data")
        assert "COALESCE(" in sql, "the JSONB ? test must be COALESCE'd: NOT (NULL ? k) is NULL"
        assert f"'{REMINDER_METADATA_KEY}'" in sql
        assert "e.data->'metadata'" in sql

    def test_every_log_derived_reader_excludes_reminders(self) -> None:
        for name, sql, col in (
            ("UNREACTED_ROWS_SQL", UNREACTED_ROWS_SQL, "e.data"),
            ("UNREACTED_ROWS_FLOORED_SQL", UNREACTED_ROWS_FLOORED_SQL, "e.data"),
            ("_INFERENCE_BEARING_PREDICATE", _INFERENCE_BEARING_PREDICATE, "data"),
        ):
            assert REMINDER_EXCLUDE_SQL.format(col=col) in sql, (
                f"{name} must exclude reminder rows via REMINDER_EXCLUDE_SQL"
            )

    def test_windower_tail_clamp_keys_on_the_newest_stimulus(self) -> None:
        src = inspect.getsource(events_mod._latest_stimulus_cumulative_tokens)
        assert "REMINDER_EXCLUDE_SQL" in src
        assert "role <> 'assistant'" in src
        clamp_src = inspect.getsource(events_mod.read_windowed_events)
        assert "_latest_stimulus_cumulative_tokens" in clamp_src

    def test_connectors_cannot_mint_a_reminder(self) -> None:
        assert REMINDER_METADATA_KEY in _RESERVED_METADATA_KEYS


class TestAppendEventReadsThePredicate:
    def test_is_reminder_evaluated_before_the_row_lock(self) -> None:
        source = textwrap.dedent(inspect.getsource(events_mod.append_event))
        tree = ast.parse(source)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef))
        call_lines = [
            n.lineno
            for n in ast.walk(func)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "is_reminder_event"
        ]
        assert call_lines, (
            "append_event must derive is_stimulus/is_user_message via is_reminder_event"
        )
        tx_lines = [
            n.lineno
            for n in ast.walk(func)
            if isinstance(n, ast.AsyncWith)
            for item in n.items
            if isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "transaction"
        ]
        assert tx_lines, "no `async with conn.transaction()` block found"
        assert all(line < min(tx_lines) for line in call_lines), (
            "the reminder predicate is pure Python and must run before the row lock"
        )
