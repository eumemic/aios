"""Unit coverage for ``_parse_form_json`` at the connector inbound boundary.

Pre-fix: ``_parse_form_json`` (``api/routers/connectors.py:95``) returned
``Any`` from ``json.loads`` with no shape validation. Both call sites
(``sender_json`` and ``metadata_json`` in ``_do_inbound`` at lines 137-138)
typed the result as ``dict[str, Any]`` but the function happily returned
lists, scalars, or even ``None`` for an explicit JSON ``null``. A
connector posting ``sender='[1,2,3]'`` to ``POST /v1/connectors/runtime/
inbound`` produced a runtime ``AttributeError`` at
``services/inbound.py:177`` (``sender.get("display_name")``) → HTTP 500.

Worse: the ``event_id`` never reached ``try_record_inbound_ack``
(the inbound flow crashes before the dedup write), so the connector's
retry mechanism would loop forever on the same poisoned event_id,
permanently freezing inbound for that chat.

Same anti-pattern as PR #446 (attachments validation): ``Any`` at the
wire boundary lets arbitrary shapes through; downstream code that
assumes ``dict`` crashes. This unit fixes the same shape at the
JSON-form-parse boundary.
"""

from __future__ import annotations

import pytest

from aios.api.routers.connectors import _parse_form_json
from aios.errors import ValidationError


class TestParseFormJson:
    def test_dict_json_passes_through(self) -> None:
        assert _parse_form_json("sender", '{"display_name": "Alice"}') == {"display_name": "Alice"}

    def test_empty_dict_json_returns_empty_dict(self) -> None:
        assert _parse_form_json("sender", "{}") == {}

    def test_none_returns_default(self) -> None:
        # None input (form field absent) → caller's default.
        assert _parse_form_json("sender", None) is None
        assert _parse_form_json("sender", None, default={}) == {}

    def test_malformed_json_raises(self) -> None:
        # Regression guard: existing JSONDecodeError branch still raises.
        with pytest.raises(ValidationError) as exc_info:
            _parse_form_json("sender", "{not valid json}")
        assert exc_info.value.detail["field"] == "sender"

    def test_json_array_rejected(self) -> None:
        """A JSON array at a dict-shaped field must raise, not pass through."""
        with pytest.raises(ValidationError) as exc_info:
            _parse_form_json("sender", "[1, 2, 3]")
        assert exc_info.value.detail["field"] == "sender"

    def test_json_scalar_rejected(self) -> None:
        """A JSON scalar (number) must raise, not pass through."""
        with pytest.raises(ValidationError) as exc_info:
            _parse_form_json("sender", "42")
        assert exc_info.value.detail["field"] == "sender"

    def test_json_string_rejected(self) -> None:
        """A JSON string (not an object) must raise, not pass through."""
        with pytest.raises(ValidationError) as exc_info:
            _parse_form_json("sender", '"just a string"')
        assert exc_info.value.detail["field"] == "sender"

    def test_json_null_rejected(self) -> None:
        """A JSON ``null`` at a dict-shaped field must raise.

        ``None`` (Python) means "field absent" and returns the default;
        ``"null"`` (JSON literal) is a wire value that decoded to None —
        the connector explicitly sent it, so it's a contract violation
        rather than absence.
        """
        with pytest.raises(ValidationError) as exc_info:
            _parse_form_json("sender", "null")
        assert exc_info.value.detail["field"] == "sender"

    def test_json_boolean_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _parse_form_json("metadata", "true")
        assert exc_info.value.detail["field"] == "metadata"
