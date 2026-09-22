from aios.harness.completion import resolved_output_reservation
from aios.harness.context_budget import (
    effective_window_max,
    output_reservation,
    served_ceiling,
)


def test_declared_served_ceiling_caps_window_and_reserves_output() -> None:
    assert served_ceiling("openai/responses/gpt-5.6-sol") == 370_000
    assert (
        effective_window_max(
            model="openai/responses/gpt-5.6-sol",
            window_max=400_000,
            params={"max_output_tokens": 32_000},
        )
        == 338_000
    )


def test_unmapped_model_preserves_window_semantics() -> None:
    assert (
        effective_window_max(
            model="some/new-model", window_max=400_000, params={"max_tokens": 32_000}
        )
        == 400_000
    )


def test_anthropic_default_is_reserved_from_window() -> None:
    model = "anthropic/claude-opus-4-1"
    reservation = resolved_output_reservation(model, None)
    # An Anthropic route must resolve a concrete reservation; ``None`` here
    # would mean the windowing exception silently fell back to window_max-only
    # semantics, which is the regression this test guards.
    assert reservation is not None
    assert reservation > 0
    assert (
        effective_window_max(
            model=model,
            window_max=200_000,
            params=None,
            output_reserve=reservation,
        )
        == 200_000 - reservation
    )


def test_anthropic_explicit_max_tokens_wins_for_windowing() -> None:
    model = "anthropic/claude-opus-4-1"
    params = {"max_tokens": 1234}
    assert resolved_output_reservation(model, params) == 1234
    assert (
        effective_window_max(
            model=model,
            window_max=200_000,
            params=params,
            output_reserve=resolved_output_reservation(model, params),
        )
        == 198_766
    )


def test_anthropic_explicit_max_output_tokens_wins_for_windowing() -> None:
    """REQUIRED TEST — windowing reserves the caller's cap, not the ceiling.

    ``output_reservation`` has always recognized ``max_output_tokens``, but
    ``default_max_tokens_for_request`` did not, so this spelling resolved to
    the model ceiling (32000 here) instead of the caller's 1234 — reserving
    ~31k tokens of context that the request was never going to spend.
    """
    model = "anthropic/claude-opus-4-1"
    params = {"max_output_tokens": 1234}
    assert resolved_output_reservation(model, params) == 1234
    assert (
        effective_window_max(
            model=model,
            window_max=200_000,
            params=params,
            output_reserve=resolved_output_reservation(model, params),
        )
        == 198_766
    )


def test_openrouter_anthropic_route_has_no_full_ceiling_reservation() -> None:
    model = "openrouter/anthropic/claude-opus-4-1"
    assert resolved_output_reservation(model, None) == 0
    assert (
        effective_window_max(
            model=model,
            window_max=200_000,
            params=None,
            output_reserve=resolved_output_reservation(model, None),
        )
        == 200_000
    )


def test_openrouter_provider_override_has_no_full_ceiling_reservation() -> None:
    model = "anthropic/claude-opus-4-1"
    params = {"custom_llm_provider": "openrouter"}
    assert resolved_output_reservation(model, params) == 0


def test_output_reservation_accepts_provider_parameter_spellings() -> None:
    assert output_reservation({"max_output_tokens": 20, "max_tokens": 10}) == 20
    assert output_reservation({"max_tokens": 10}) == 10
    assert output_reservation({"max_completion_tokens": 30}) == 30
    assert output_reservation(None) == 0


def test_overflow_shrink_is_applied_after_ceiling() -> None:
    # (b) A mapped-ceiling model shrinks off the ceiling-derived input cap.
    assert (
        effective_window_max(
            model="openai/responses/gpt-5.6-sol",
            window_max=400_000,
            params={"max_output_tokens": 20_000},
            shrink_factor=0.8,
        )
        == 280_000
    )


def test_unmapped_model_overflow_retry_shrinks_below_window_max() -> None:
    """(a) Regression for the 2026-07-09 outage class: a model with no declared
    ceiling MUST still tighten its budget on an overflow retry (shrink_factor <
    1). Before the fix the unmapped branch returned ``window_max`` verbatim, so
    the overflow retry re-sent the identical oversized request and looped up the
    reschedule ladder — strictly worse than pre-PR behavior (which terminated).
    """
    full = effective_window_max(
        model="some/new-model", window_max=400_000, params={"max_tokens": 32_000}
    )
    shrunk = effective_window_max(
        model="some/new-model",
        window_max=400_000,
        params={"max_tokens": 32_000},
        shrink_factor=0.8,
    )
    assert full == 400_000  # full budget preserved when not retrying
    assert shrunk == 320_000  # 0.8 * window_max
    assert shrunk < full  # the retry is STRICTLY smaller, never verbatim


def test_unmapped_model_overflow_shrink_is_progressive() -> None:
    """Consecutive unmapped overflows tighten further each time (0.8, 0.64),
    so no two retries carry the identical budget."""
    first = effective_window_max(
        model="some/new-model", window_max=400_000, params=None, shrink_factor=0.8
    )
    second = effective_window_max(
        model="some/new-model", window_max=400_000, params=None, shrink_factor=0.64
    )
    assert first == 320_000
    assert second == 256_000
    assert second < first
