from typing import Any

from aios.harness.completion import resolved_context_limit, resolved_output_reservation
from aios.harness.context_budget import (
    effective_window_max,
    output_reservation,
    served_ceiling,
)


def request_budget(
    *,
    model: str,
    window_max: int,
    params: dict[str, Any] | None = None,
    shrink_factor: float = 1.0,
) -> int:
    """Form the budget exactly as ``loop.py`` does, resolvers included.

    The route-resolved reservation and limit are the whole point of the
    ``completion``/``context_budget`` seam, so the tests below drive the
    composition rather than hand-picked arguments — a resolver that stops
    agreeing with the injection gate fails here instead of drifting quietly.
    """
    return effective_window_max(
        model=model,
        window_max=window_max,
        params=params,
        output_reserve=resolved_output_reservation(model, params),
        context_limit=resolved_context_limit(model, params),
        shrink_factor=shrink_factor,
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
    """An Anthropic-shaped route with no caller cap reserves the ceiling the
    harness is about to inject, so the prompt plus that reservation cannot
    exceed the limit they are jointly charged against.

    Asserted RELATIVE to the resolvers: both numbers come from LiteLLM's
    capability map, which differs between the bundled backup and the fetched
    remote map (unit tests block egress and get the backup), so pinning
    literals here would pin the catalog snapshot instead of the behavior.
    """
    model = "anthropic/claude-opus-4-1"
    reservation = resolved_output_reservation(model, None)
    limit = resolved_context_limit(model, None)
    assert reservation is not None and reservation > 0
    assert limit is not None and limit > reservation
    # A window_max set AT the provider limit is the case the reservation exists
    # for: a full-window prompt plus the injected cap would breach the limit,
    # so the input cap gives way by exactly the reservation.
    assert request_budget(model=model, window_max=limit) == limit - reservation


def test_anthropic_reservation_never_shrinks_a_window_that_already_fits() -> None:
    """The reservation binds the PROVIDER's ceiling, never the operator's
    ``window_max``.

    Regression for the review finding on ``13e8d003``: that revision computed
    ``max(1, window_max - reservation)``, which (a) silently cost every agent
    its model's output ceiling in history even when the limit had room to spare
    and (b) collapsed the budget to 1 for any agent whose ``window_max`` was at
    or below that ceiling — ``window_max`` only validates ``ge=1`` — at which
    point ``read_windowed_events`` raises "no budget remains for events" and the
    step hard-fails outright. Both legs are asserted here.
    """
    model = "anthropic/claude-opus-4-1"
    reservation = resolved_output_reservation(model, None)
    limit = resolved_context_limit(model, None)
    assert reservation is not None and limit is not None

    # (a) Any window that already leaves the reservation room inside the limit
    # is handed through verbatim — no silent loss of configured history.
    fits = limit - reservation
    assert request_budget(model=model, window_max=fits) == fits
    assert request_budget(model=model, window_max=fits - 10_000) == fits - 10_000

    # (b) A window at or below the model's own output ceiling keeps its full
    # budget. Under the 13e8d003 formula every one of these returned 1.
    for window_max in (1_000, reservation // 2, reservation, reservation + 500):
        assert request_budget(model=model, window_max=window_max) == window_max


def test_anthropic_explicit_max_tokens_wins_for_windowing() -> None:
    model = "anthropic/claude-opus-4-1"
    params: dict[str, Any] = {"max_tokens": 1234}
    limit = resolved_context_limit(model, params)
    assert resolved_output_reservation(model, params) == 1234
    assert limit is not None
    assert request_budget(model=model, window_max=limit, params=params) == limit - 1234


def test_anthropic_explicit_max_output_tokens_wins_for_windowing() -> None:
    """Windowing reserves the caller's cap, not the output ceiling."""
    model = "anthropic/claude-opus-4-1"
    params = {"max_output_tokens": 1234}
    limit = resolved_context_limit(model, params)
    assert resolved_output_reservation(model, params) == 1234
    assert limit is not None
    assert request_budget(model=model, window_max=limit, params=params) == limit - 1234


def test_openrouter_anthropic_route_has_no_full_ceiling_reservation() -> None:
    model = "openrouter/anthropic/claude-opus-4-1"
    assert resolved_output_reservation(model, None) == 0
    # No injected cap on this route, so nothing to reserve and no limit to bind
    # against: the budget is the operator's window_max verbatim.
    assert resolved_context_limit(model, None) is None
    assert request_budget(model=model, window_max=200_000) == 200_000


def test_openrouter_provider_override_has_no_full_ceiling_reservation() -> None:
    model = "anthropic/claude-opus-4-1"
    params: dict[str, Any] = {"custom_llm_provider": "openrouter"}
    assert resolved_output_reservation(model, params) == 0
    assert resolved_context_limit(model, params) is None
    assert request_budget(model=model, window_max=200_000, params=params) == 200_000


def test_anthropic_overflow_retry_still_shrinks_off_the_reserved_cap() -> None:
    """The shrink ladder composes with the reservation rather than bypassing
    it: a retry is strictly smaller than the already-reserved budget."""
    model = "anthropic/claude-opus-4-1"
    limit = resolved_context_limit(model, None)
    reservation = resolved_output_reservation(model, None)
    assert limit is not None and reservation is not None
    full = request_budget(model=model, window_max=limit)
    shrunk = request_budget(model=model, window_max=limit, shrink_factor=0.8)
    assert full == limit - reservation
    assert shrunk == int(full * 0.8)
    assert shrunk < full


def test_non_anthropic_routes_keep_window_only_budget() -> None:
    """An OpenAI-shaped route sizes its own output against the remaining
    window, so it resolves no reservation and no limit — unchanged behavior."""
    model = "openai/gpt-4.1"
    assert resolved_output_reservation(model, None) is None
    assert resolved_context_limit(model, None) is None
    assert request_budget(model=model, window_max=400_000) == 400_000


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
