"""Tests for cost extraction in ``aios.harness.completion``."""

from __future__ import annotations

from types import SimpleNamespace

from aios.harness.completion import _extract_cost


class TestExtractCost:
    def test_returns_float_when_hidden_params_has_response_cost(self) -> None:
        response = SimpleNamespace(_hidden_params={"response_cost": 0.00342})
        assert _extract_cost(response) == 0.00342

    def test_returns_none_when_hidden_params_attribute_missing(self) -> None:
        assert _extract_cost(SimpleNamespace()) is None

    def test_returns_none_when_hidden_params_is_none(self) -> None:
        response = SimpleNamespace(_hidden_params=None)
        assert _extract_cost(response) is None

    def test_returns_none_when_response_cost_key_missing(self) -> None:
        response = SimpleNamespace(_hidden_params={})
        assert _extract_cost(response) is None

    def test_returns_none_when_response_cost_is_none(self) -> None:
        response = SimpleNamespace(_hidden_params={"response_cost": None})
        assert _extract_cost(response) is None

    def test_returns_none_for_plain_dict_response(self) -> None:
        """Plain dicts (used in e2e harness mocks) have no _hidden_params attr."""
        assert _extract_cost({"choices": [], "usage": {}}) is None

    def test_coerces_int_cost_to_float(self) -> None:
        response = SimpleNamespace(_hidden_params={"response_cost": 1})
        result = _extract_cost(response)
        assert result == 1.0
        assert isinstance(result, float)
