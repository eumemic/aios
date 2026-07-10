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
    assert (
        effective_window_max(
            model="openai/responses/gpt-5.6-sol",
            window_max=400_000,
            params={"max_output_tokens": 20_000},
            shrink_factor=0.8,
        )
        == 280_000
    )
