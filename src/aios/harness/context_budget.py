"""O(1) request-budget formation from metadata and already-known parameters."""

from __future__ import annotations

from typing import Any

# Served ceilings can differ materially from public model-card context windows.
# Unknown models deliberately have no entry: they retain window_max-only behavior.
_SERVED_CEILINGS: dict[str, int] = {
    "openai/responses/gpt-5.6-sol": 370_000,
}


def served_ceiling(model: str) -> int | None:
    """Return the empirically served input+output ceiling, when declared."""
    return _SERVED_CEILINGS.get(model)


def output_reservation(params: dict[str, Any] | None) -> int:
    """Return the request's explicit maximum output/reasoning reservation."""
    if not params:
        return 0
    for key in ("max_output_tokens", "max_tokens", "max_completion_tokens"):
        value = params.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    thinking = params.get("thinking")
    if isinstance(thinking, dict):
        value = thinking.get("budget_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def effective_window_max(
    *,
    model: str,
    window_max: int,
    params: dict[str, Any] | None,
    output_reserve: int | None = None,
    context_limit: int | None = None,
    shrink_factor: float = 1.0,
) -> int:
    """Form the total input budget before the windower subtracts class masses.

    The caller passes the result to ``read_windowed_events``; that function
    subtracts its already-computed system/tools/current-turn reserves using the
    calibrated class coefficients. No assembled-context token pass is involved.

    **The output reservation is only ever subtracted from a CEILING, never from
    ``window_max``.** The two are different quantities: ``window_max`` is the
    operator's own input budget for the agent, while the ceiling is the total
    the provider charges ``input + max_tokens`` against. Subtracting the
    reservation from ``window_max`` would silently shrink every agent's
    configured history by the model's output ceiling, and for any agent whose
    ``window_max`` is at or below that ceiling (``window_max`` only validates
    ``ge=1``) it collapses the budget to 1 — at which point
    ``read_windowed_events`` raises ``"no budget remains for events"`` and the
    step hard-fails. ``min(window_max, ceiling - reservation)`` is
    correct-by-construction instead: the operator's cap and the provider's cap
    each bind independently, and neither is corrupted by the other.

    ``context_limit`` supplies that ceiling for routes that have no entry in
    :data:`_SERVED_CEILINGS` but whose limit the caller can resolve from model
    metadata (see :func:`~aios.harness.completion.resolved_context_limit` —
    Anthropic-shaped routes, where ``input + max_tokens`` is charged against one
    limit). ``served_ceiling`` still wins when declared: an empirically measured
    served ceiling outranks a published model-card number.
    """
    ceiling = served_ceiling(model)
    if ceiling is None:
        ceiling = context_limit
    reservation = output_reservation(params) if output_reserve is None else output_reserve
    if ceiling is None:
        # Unmapped models retain window_max-only behavior at full budget
        # (shrink_factor == 1.0, today's semantics). But an overflow retry
        # (shrink_factor < 1) MUST still tighten the budget here — otherwise the
        # retry re-sends the identical oversized request and loops verbatim up
        # the reschedule ladder (the 2026-07-09 Ultron/sol outage class).
        return max(1, int(window_max * shrink_factor))
    input_cap = min(window_max, max(1, ceiling - reservation))
    return max(1, int(input_cap * shrink_factor))
