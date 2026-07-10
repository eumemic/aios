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


def test_output_reservation_accepts_provider_parameter_spellings() -> None:
    assert output_reservation({"max_output_tokens": 20, "max_tokens": 10}) == 20
    assert output_reservation({"max_tokens": 10}) == 10
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
