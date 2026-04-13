"""Unit tests for token usage normalization."""

from __future__ import annotations

from aios.harness.completion import _normalize_usage


class TestNormalizeUsage:
    """Tests for _normalize_usage which maps LiteLLM fields to our names."""

    def test_standard_openai_fields(self) -> None:
        raw = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def test_anthropic_cache_fields(self) -> None:
        """Anthropic passes cache fields at the top level via LiteLLM."""
        raw = {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 30,
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 200,
            "output_tokens": 80,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 30,
        }

    def test_openai_cached_tokens_in_details(self) -> None:
        """OpenAI puts cache reads in prompt_tokens_details.cached_tokens."""
        raw = {
            "prompt_tokens": 300,
            "completion_tokens": 100,
            "total_tokens": 400,
            "prompt_tokens_details": {"cached_tokens": 120},
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 300,
            "output_tokens": 100,
            "cache_read_input_tokens": 120,
            "cache_creation_input_tokens": 0,
        }

    def test_anthropic_cache_read_takes_precedence(self) -> None:
        """Top-level cache_read_input_tokens wins over prompt_tokens_details."""
        raw = {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "cache_read_input_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 999},
        }
        result = _normalize_usage(raw)
        assert result["cache_read_input_tokens"] == 50

    def test_empty_dict(self) -> None:
        result = _normalize_usage({})
        assert result == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def test_none_values_treated_as_zero(self) -> None:
        raw = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "prompt_tokens_details": None,
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def test_zero_values_preserved(self) -> None:
        raw = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
